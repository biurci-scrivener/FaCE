#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"
#include "polyscope/pick.h"
#include "args/args.hxx"
#include "io.h"
#include "pc.h"
#include "vis.h"
#include "faraday.h"
#include "solve.h"

#include <igl/grad.h>
#include <igl/barycenter.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/volume.h>
#include <igl/octree.h>
#include <igl/knn.h>

#include "imgui.h"
#include <string>

bool USE_BILAPLACIAN = false;
bool SAVE_OUTFILE = false;
bool RUN_HEADLESS = false;
bool PRESERVE_INTERIOR_POINTS = true;
std::string outfile_path = "";
std::string filename = "";
static char outfile_polyscope[256] = "output.ply";

struct Faraday f;
bool PICKING_ENABLED = false;
int SCULPT_MODE = 0; // 0 is carve, 1 is shield
bool key_s_just_pressed = false;

/*
	true if the user has added new cage locations,
	which is the only reason the "base solve" would need to be run again
*/
bool dirs_solve_is_stale = false;
int radio_iso_choice = 0;
int radio_filter_choice = 0;
int u_idx = 0;
int pt_idx = 0;
double cutoff = 1e-6;
double isosurface_param;
double vis_isosurface_param = 1;
double usr_alpha = 0.01;
double charge_scale = 5.;

const char * scuplt_strings = "Carve\0Shield\0";

void isoSurfaceUpdate(struct Faraday &f_) {
	fillTetHoles(f_);
	extract_isosurface(f_, vis_isosurface_param);
	vis_isosurface(f_);
}

void runSolve(struct Faraday &f_, std::vector<int> &pt_constraints) {
	if (dirs_solve_is_stale) {
		setCageVerts(f_);
		solvePotentialOverDirs(f_);
	}

	solvePotentialPointCharges(f_, pt_constraints, charge_scale);
	solveMaxFunction(f_);
	estimateNormals(f_);
}

void runSolve(struct Faraday &f_) {
	solvePotentialOverDirs(f_);
	solveMaxFunction(f_);
	estimateNormals(f_);
}

void runPipeline(struct Faraday &f_) {
	/*
		Does everything to tetrahedralize the domain, solve for the potentials etc.
		Only the f_.P atrribute needs to be populated before calling this function.
		By the end, pretty much every other attribute will have been filled in
	*/
	std::cout << "Computing octree (input points)" << std::endl;
	igl::octree(f_.P, f_.PI, f_.CH, f_.CN, f_.W);
	std::cout << "Computing KNN" << std::endl;
	igl::knn(f_.P, 2, f_.PI, f_.CH, f_.CN, f_.W, f_.knn);
	std::cout << "Computing cage radii" << std::endl;
	computeNearestNeighborDists(f_);

	prepareTetgen(f_);

	std::cout << "Starting tetrahedralization..." << std::endl;
	std::cout << "=======================" << std::endl;
	double tet_area = 0.001 * (f_.bb.row(0) - f_.bb.row(1)).cwiseAbs().prod();

	if (igl::copyleft::tetgen::tetrahedralize(	f_.V, f_.F, f_.H, f_.VM, f_.FM, f_.R, "pq1.414a"+ std::to_string(tet_area),
												f_.TV, f_.TT, f_.TF, f_.TM, f_.TR, f_.TN, f_.PT, f_.FT, f_.num_regions)) exit(-1);

	std::cout << "=======================" << std::endl;
	std::cout << "Finished tetrahedralizing" << std::endl;


	std::cout << "Computing octree (tet. vertices)" << std::endl;
	igl::octree(f_.TV, f_.PI_TV, f_.CH_TV, f_.CN_TV, f_.W_TV);


	std::cout << "Computing cell barycenters" << std::endl;
	igl::barycenter(f_.TV, f_.TT, f_.BC);
	std::cout << "Computing grad operator, cell volumes" << std::endl;
	faster_igl_quantities(f_);
	std::cout << "Computing Laplacian" << std::endl;
	igl::cotmatrix(f_.TV, f_.TT, f_.M);
	f_.L = f_.M;
	if (USE_BILAPLACIAN) {
		std::cout << "\tComputing Bilaplacian" << std::endl;
		f_.L = f_.L * f_.L;
	} 
	std::cout << "Assigning boundary/cage verts." << std::endl;
	findBdryCage(f_);
	std::cout << "Assigning tets." << std::endl;
	findTets(f_);
	std::cout << "Computing f_to_v matrix" << std::endl;
	build_f_to_v_matrix(f_);

	// solve for field over many directions

	runSolve(f_);
	computePointsetFunctions(f_);

	// extract an isosurface

	std::cout << "Extracting isosurface: filling tet mesh holes" << std::endl;

	fillTetHoles(f_);

	extract_isosurface(f_, vis_isosurface_param);
	classifyInterior(f_, isosurface_param);


	std::cout << "=======================" << std::endl << std::endl;

}

void myCallback() {
	ImGui::PushItemWidth(100);
	bool sculpt_just_changed = false;

	if (ImGui::CollapsingHeader("Interior point classification", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Text("Alpha used for interior point filtering:");

		if (ImGui::RadioButton("Dense input (alpha = 0.05)", radio_iso_choice == 0)) {
			radio_iso_choice = 0;
			isosurface_param = 0.05;
		}
		if (ImGui::RadioButton("Sparse input (large gaps, alpha = 0.01)", radio_iso_choice == 1)) {
			radio_iso_choice = 1;
			isosurface_param = 0.01;
		}
		if (ImGui::RadioButton("Custom alpha:", radio_iso_choice == 2)) {
			radio_iso_choice = 2;
			isosurface_param = usr_alpha;
		}
		if (ImGui::InputDouble("##Usr_alpha", &usr_alpha, 0.01, 0.1, "%.6f")) {
			 usr_alpha = usr_alpha < 0. ? 0. : usr_alpha;
			if (radio_iso_choice == 2) {
				isosurface_param = usr_alpha;
			}
		}
		if (ImGui::Button("Reclassify interior")) {
			classifyInterior(f, isosurface_param);
			vis_interior(f);
		}
	
		if (SAVE_OUTFILE) {
			if (ImGui::Button("Save now (filtered)")) {
				savePLY(f, outfile_path, false);
			}
		
			if (ImGui::Button("Save now (keep interior)")) {
				savePLY(f, outfile_path, true);
			}
		}	
	}

	if (ImGui::CollapsingHeader("Faraday sculpting", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Checkbox("Enable user sculpting tool", &PICKING_ENABLED)) {
			sculpt_just_changed = true;
		};

		if (ImGui::Combo("##Sculpting Mode", &SCULPT_MODE, scuplt_strings, 2)) {
			sculpt_just_changed = true;
		};
		ImGui::SameLine();
		if (ImGui::Button("Recompute")) {

			runSolve(f, f.point_charges);

			computePointsetFunctions(f);
			fillTetHoles(f);
			extract_isosurface(f, vis_isosurface_param);
			classifyInterior(f, isosurface_param);
			
			if (dirs_solve_is_stale) {
				reInitCageStructures(f);
				dirs_solve_is_stale = false;
			}

			vis_normal_results(f);
			vis_max(f);
			vis_isosurface(f);
			vis_interior(f);
			std::cout << "Done" << std::endl;
		}
		ImGui::SameLine();
		if (ImGui::Button("Clear selected")) {
			if (SCULPT_MODE == 0) {
				std::cout << "Clearing user's point charges" << std::endl;
				polyscope::removeStructure("User's point charges");
				f.point_charges.clear();
				sculpt_just_changed = true;
			} else {
				std::cout << "Clearing user's cage locations" << std::endl;
				polyscope::removeStructure("User's cage locations");
				f.user_cages.clear();
				dirs_solve_is_stale = true;
				sculpt_just_changed = true;
			}
			
		}

		ImGui::Text("");
		ImGui::TextWrapped("Isosurface parameter. Clicking on this \"proxy surface\" creates a new point charge (carving mode) or cage vertex (shielding mode).");
		ImGui::SetNextItemWidth(200.0f);
		if (ImGui::InputDouble("##Iso", &vis_isosurface_param, 0.01, 0.1, "%.6f")) {
			vis_isosurface_param = vis_isosurface_param < 0. ? 0. : vis_isosurface_param;
		}
		if (ImGui::Button("Update isosurface")) {
			isoSurfaceUpdate(f);
		}

	}

	if (ImGui::CollapsingHeader("Save results", ImGuiTreeNodeFlags_DefaultOpen)) {

		ImGui::TextWrapped("Results will be saved in the current terminal directory (PLY format). Choose below whether or not to perform interior point filtering: this is needed only for inputs with interior structures.");

		ImGui::InputText("##OutFile", outfile_polyscope, IM_ARRAYSIZE(outfile_polyscope));

		if (ImGui::RadioButton("Don't filter interior points", radio_filter_choice == 0)) {
			radio_filter_choice = 0;
		}
		if (ImGui::RadioButton("Filter interior points with alpha (set at top of UI)", radio_filter_choice == 1)) {
			radio_filter_choice = 1;
		}

		if (ImGui::Button("Save (WARNING: can overwrite existing files!)")) {
			savePLY(f, outfile_polyscope, !radio_filter_choice);
		}
	}

	if (ImGui::CollapsingHeader("Debug")) {
		ImGui::Text("Add point charge at: ");
		ImGui::SameLine();
		ImGui::InputInt("##PtCharge", &pt_idx);
		if (ImGui::Button("Add")) {
			addPointCharge(f, pt_idx);
			if (SCULPT_MODE == 0) {
				vis_user_charge_locations(f);
			} else {
				vis_user_cage_locations(f);
			}
		}
		if (ImGui::InputDouble("##PC_param", &charge_scale, 0.01, 0.1, "%.6f")) {
			charge_scale = charge_scale < 0. ? 0. : charge_scale;
		}

		ImGui::Text("");

		ImGui::Text("View direction: ");
		ImGui::SameLine();
		if (ImGui::InputInt("##Dir", &u_idx)) {
			if (u_idx < 0) {
				u_idx = 0;
			} else if (u_idx >= f.u.cols()) {
				u_idx = f.u.cols() - 1;
			}
		}
		ImGui::SameLine();
		if (ImGui::Button("Update")) {
			vis_u(f, u_idx);
		}
		if (ImGui::Button("View gradient magnitude of this rank")) {
			std::cout << "Viewing rank " << u_idx << std::endl;
			vis_grad_ranked(f, u_idx);
		}


		std::string list_pt_charges = "Currently, point charges at: \n";
		for (int pt: f.point_charges) list_pt_charges += std::to_string(pt) + "\n";
		ImGui::Text("%s", list_pt_charges.c_str());
	}


	ImGui::PopItemWidth();


	// Handle IO (mouse events, keys)
	ImGuiIO& io = ImGui::GetIO();
	if ((io.MouseClicked[0]) && (PICKING_ENABLED) && (!sculpt_just_changed)) { // if the left mouse button was clicked
		// gather values
		glm::vec2 screenCoords{io.MousePos.x, io.MousePos.y};
		std::pair<polyscope::Structure*, size_t> pickPair = 
			polyscope::pick::pickAtScreenCoords(screenCoords);

		// figure out what type of object was clicked
		polyscope::Structure * structure_clicked;
		int index_clicked;
		int index_unpacked;
		bool is_vertex;
		std::tie(structure_clicked, index_clicked) = pickPair;

		if (!(structure_clicked == nullptr) && (structure_clicked->getName() == "Isosurface")) {
			auto iso_polyscope = polyscope::getSurfaceMesh("Isosurface");

			int nVertices = iso_polyscope->nVertices();
			int nFaces = iso_polyscope->nFaces();
			int nEdges = iso_polyscope->nEdges();
			int nHalfedges = iso_polyscope->nHalfedges();
			int nCorners = iso_polyscope->nCorners();

			if (index_clicked < nVertices) {
				is_vertex = true;
				index_unpacked = index_clicked;
				std::cout << "Clicked vertex " << index_unpacked << " of the isosurface" << std::endl;
			} else if (index_clicked < nVertices + nFaces) {
				is_vertex = false;
				index_unpacked = index_clicked - nVertices;
				std::cout << "Clicked face " << index_unpacked << " of the isosurface" << std::endl;
			} else {
				// I will not be entertaining this. Click again buddy
				return;
			}
			
			if (SCULPT_MODE == 0) {
				addPointCharge(f, getQueryPointFromClicked(f, index_unpacked, is_vertex));
				vis_user_charge_locations(f);
			} else {
				addUserCage(f, getQueryPointFromClicked(f, index_unpacked, is_vertex));
				dirs_solve_is_stale = true;
				vis_user_cage_locations(f);
			}
			
		}
	}

	if (ImGui::IsKeyDown(ImGuiKey_S)) {
		if (!key_s_just_pressed) {
			PICKING_ENABLED = !PICKING_ENABLED;
			key_s_just_pressed = true;
			auto vis_iso = polyscope::getSurfaceMesh("Isosurface");
			vis_iso->setEnabled(PICKING_ENABLED);
		}
	} else {
		key_s_just_pressed = false;
	} 

	if (ImGui::IsKeyDown(ImGuiKey_C)) {
		SCULPT_MODE = 0;
	} else if (ImGui::IsKeyDown(ImGuiKey_V)) {
		SCULPT_MODE = 1;
	}
}

int main(int argc, char **argv) {

	// Configure the argument parser
	args::ArgumentParser parser("Implementation of FaCE.\nUse --h along with additional command line arguments to run FaCE in headless mode, which bypasses the Polyscope visualization.");
	args::Positional<std::string> inputFilename(parser, "pc", "Location of input point cloud");
	args::Group group(parser);
	args::Flag headless(group, "headless", "Run in headless mode (no Polyscope vis., useful for scripting)", {"h"});
	args::Flag doInteriorF(group, "interior", "Perform filtering of interior points (if running headless)", {"i"});
	args::ValueFlag<std::string> outputFilename(parser, "", "Location in which to save output (if running headless)", {"o"});
	args::ValueFlag<std::string> isoVal(parser, "", "Alpha threshold for interior point classification (if running headless)", {"a"});
	args::Flag bilaplacian(group, "bilaplacian", "Use the Bilaplacian instead of the Laplacian", {"b"});

	// Parse args
	try {
	parser.ParseCLI(argc, argv);
	} catch (args::Help &h) {
	std::cout << parser;
	return 0;
	} catch (args::ParseError &e) {
	std::cerr << e.what() << std::endl;
	std::cerr << parser;
	return 1;
	}

	if (!inputFilename) {
		std::cerr << "Please specify an input file (.xyz, .ply, .obj)." << std::endl;
		return EXIT_FAILURE;
	} else {
		filename = args::get(inputFilename);
	}
	
	RUN_HEADLESS = headless;

	if (!isoVal) {
		radio_iso_choice = 0;
		isosurface_param = 0.05;
	} else {
		radio_iso_choice = 2;
		isosurface_param = atof(args::get(isoVal).c_str());
		usr_alpha = isosurface_param;
	}

	if (!RUN_HEADLESS) {
		std::cout << "Running with Polyscope GUI" << std::endl;
	} else {
		if (!outputFilename) {
			std::cerr << "Please specify a location for the output with --o" << std::endl;
			return EXIT_FAILURE;
		} else {
			SAVE_OUTFILE = true;
			outfile_path = args::get(outputFilename);
		}
	 
		if (doInteriorF) {
			PRESERVE_INTERIOR_POINTS = false;
			std::cout << "Interior points will be filtered." << std::endl;
			std::cout << "\tAlpha = " << isosurface_param;
			if (isoVal) {
				std::cout << " (specified by user)" << std::endl;
			} else {
				std::cout << " (default for dense inputs)" << std::endl;
			}
		} else {
			std::cout << "Interior points will not be filtered." << std::endl;
		}
	}

	if (bilaplacian) {
		USE_BILAPLACIAN = true;
		std::cout << "Using Bilaplacian" << std::endl;
	}
	
	// load point cloud 

	std::tie(f.P, f.N) = my_load_points(filename);

	std::cout << "Loaded file " << filename << " with " << f.P.rows() << " points." << std::endl;

	std::cout << "=======================" << std::endl;

	runPipeline(f);

	if (!RUN_HEADLESS) {
		polyscope::init();
		polyscope::options::uiScale = 1.0;
		polyscope::state::userCallback = myCallback;

		initPolyscopeStructures(f);

		polyscope::show();
	} else if (SAVE_OUTFILE) {
		savePLY(f, outfile_path, PRESERVE_INTERIOR_POINTS);
	}

	return EXIT_SUCCESS;

}