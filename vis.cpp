#include "vis.h"


void initPolyscopeStructures(struct Faraday &f) {
	// initialize structures

	auto pc = polyscope::registerPointCloud("Points", f.P);
	pc->setEnabled(true);
	pc->addVectorQuantity("Normals, true", f.N);
    Eigen::VectorXi pt_id = Eigen::VectorXi::LinSpaced(f.P.rows(), 0, f.P.rows() - 1);
    pc->addScalarQuantity("Id", pt_id);

    auto v_cage = polyscope::registerPointCloud("Cage vertices", f.V_cage);
    v_cage->setEnabled(false);

	auto tet_mesh = polyscope::registerTetMesh("Tet. mesh", f.TV, f.TT);
    tet_mesh->setEnabled(false);
	tet_mesh->setCullWholeElements(false);

    auto tet_mesh_filled = polyscope::registerTetMesh("Tet. mesh (filled)", f.TV_filled, f.TT_filled);
    tet_mesh_filled->setEnabled(false);
    tet_mesh_filled->setCullWholeElements(false);

    if ((f.TV.rows() < 2000000)) {
        // polyscope will probably crash otherwise
        auto tet_slice = polyscope::addSceneSlicePlane();
        tet_slice->setDrawPlane(false);
        tet_slice->setDrawWidget(true);
        tet_slice->setVolumeMeshToInspect("Tet. mesh");
    }
    
	auto tv_vis = polyscope::registerPointCloud("Tet. mesh, vertices", f.TV);
    tv_vis->addScalarQuantity("Bdry.", f.is_bdry_tv);
	tv_vis->addScalarQuantity("Cage", f.is_cage_tv);
	tv_vis->setPointRenderMode(polyscope::PointRenderMode::Quad);
	tv_vis->setEnabled(false);

	auto bc_vis = polyscope::registerPointCloud("Tet. mesh, cell centers", f.BC);
	bc_vis->setPointRenderMode(polyscope::PointRenderMode::Quad);
	bc_vis->setEnabled(false);
    
	// call vis. functions

    vis_normal_results(f);
    vis_isosurface(f);

    auto iso_vis = polyscope::getSurfaceMesh("Isosurface");
    iso_vis->setEnabled(false);

    vis_interior(f);
	vis_max(f);
    
}

void reInitCageStructures(struct Faraday &f) {

    polyscope::PointCloud * tv_vis = polyscope::getPointCloud("Tet. mesh, vertices");
    tv_vis->addScalarQuantity("Cage", f.is_cage_tv);

    f.V_cage = Eigen::MatrixXd(f.is_cage_tv.sum(), 3);

    int cage_idx = 0;
    for (int i = 0; i < f.TV.rows(); i++) {
        if (f.is_cage_tv(i)) {
            f.V_cage.row(cage_idx) = f.TV.row(i);
            cage_idx++;
        }
    }

    auto v_cage = polyscope::registerPointCloud("Cage vertices", f.V_cage);
    v_cage->setEnabled(false);

}

void vis_user_cage_locations(struct Faraday &f) {
    if (f.user_cages.empty()) {
        return;
    }

    Eigen::MatrixXd points(f.user_cages.size(), 3);
    Eigen::VectorXd point_ids(f.user_cages.size());
    Eigen::VectorXd point_vals(f.user_cages.size());

    int i = 0;
    for (int c: f.user_cages) {
        points.row(i) = f.TV.row(c); 
        i++;
    }

    auto user_cage_vis = polyscope::registerPointCloud("User's cage locations", points);
    user_cage_vis->setPointColor({0.,1.,0.});
}

void vis_user_charge_locations(struct Faraday &f) {
    if (f.point_charges.empty()) {
        return;
    }

    Eigen::MatrixXd points(f.point_charges.size(), 3);
    Eigen::VectorXd point_ids(f.point_charges.size());
    Eigen::VectorXd point_vals(f.point_charges.size());

    int i = 0;
    for (int c: f.point_charges) {
        points.row(i) = f.TV.row(c); 
        i++;
    }

    auto point_charge_vis = polyscope::registerPointCloud("User's point charges", points);
    point_charge_vis->setPointColor({1.,0.,0.});
}

void vis_normal_results(struct Faraday &f) {
    polyscope::PointCloud * pc = polyscope::getPointCloud("Points");
    auto vis_est = pc->addVectorQuantity("Normals, estimated", f.N_est);
    vis_est->setEnabled(true);
}

void vis_isosurface(struct Faraday &f) {
    auto sf_vis = polyscope::registerSurfaceMesh("Isosurface", f.SV, f.SF);
}

void vis_interior(struct Faraday &f) {
    polyscope::PointCloud * pc = polyscope::getPointCloud("Points");
    pc->addScalarQuantity("Interior", f.is_interior);
    pc->addScalarQuantity("WN", f.wn);
}

void vis_grad_ranked(struct Faraday &f, int idx) {

    std::cout << "\tViewing rank " << idx << std::endl;

    polyscope::VolumeMesh * tet_mesh = polyscope::getVolumeMesh("Tet. mesh");
    auto vis = tet_mesh->addVertexScalarQuantity("Grad. mag. of specified rank", f.gradmag_sorted.col(idx));
    vis->setEnabled(true);

    polyscope::PointCloud * tet_pc = polyscope::getPointCloud("Tet. mesh, vertices");
    auto vis_p = tet_pc->addScalarQuantity("Grad. mag. of specified rank", f.gradmag_sorted.col(idx));
    vis_p->setEnabled(true);

}


void vis_u(struct Faraday &f, int idx) {
    polyscope::VolumeMesh * tet_mesh = polyscope::getVolumeMesh("Tet. mesh");
    auto u_gm = tet_mesh->addVertexScalarQuantity("u, grad mag.", f.gradmag.col(idx));
    auto u = tet_mesh->addVertexScalarQuantity("u", f.u.col(idx));
    u_gm->setEnabled(true);

    polyscope::PointCloud * tet_pc = polyscope::getPointCloud("Tet. mesh, vertices");
    tet_pc->addScalarQuantity("u", f.u.col(idx));
    tet_pc->addVectorQuantity("u, grad.", f.u_grad.middleCols(idx * 3, 3));
    tet_pc->addScalarQuantity("u, grad mag.", f.gradmag.col(idx));
}

void vis_max(struct Faraday &f) {

    polyscope::PointCloud * pc = polyscope::getPointCloud("Points"); 
    pc->addScalarQuantity("E_max, average across cage", f.P_avg_value);
    pc->addScalarQuantity("E_max, largest val. across cage", f.P_max);
    pc->addScalarQuantity("E_max, variance across cage", f.P_var);
    pc->addScalarQuantity("||\\nabla E_max||", f.P_max_grad_norm);

    polyscope::VolumeMesh * tet_mesh = polyscope::getVolumeMesh("Tet. mesh");
    auto quant1 = tet_mesh->addVertexScalarQuantity("E_max",  f.max);
    quant1->setEnabled(true);
    tet_mesh->addVertexScalarQuantity("||\\nabla E_max||", f.max_grad_norm);

    polyscope::VolumeMesh * tet_mesh_filled = polyscope::getVolumeMesh("Tet. mesh (filled)");
    auto quant2 = tet_mesh_filled->addVertexScalarQuantity("E_max", f.max_filled);
    quant2->setEnabled(true);

    polyscope::PointCloud * tet_pc = polyscope::getPointCloud("Tet. mesh, vertices");
    tet_pc->addScalarQuantity("E_max", f.max);
    tet_pc->addVectorQuantity("\\nabla E_max", f.max_grad);
    tet_pc->addVectorQuantity("\\nabla E_max, normalized", f.max_grad_normalized);

    polyscope::PointCloud * tet_bc = polyscope::getPointCloud("Tet. mesh, cell centers");
    tet_bc->addVectorQuantity("\\nabla E_max", f.max_grad_tets);
    tet_bc->addVectorQuantity("\\nabla E_max, normalized", f.max_grad_tets_normalized);

    polyscope::PointCloud * v_cage = polyscope::getPointCloud("Cage vertices");
    Eigen::MatrixXd max_grad_cage = Eigen::MatrixXd::Zero(f.is_cage_tv.sum(), 3);
    Eigen::MatrixXd max_grad_normalized_cage = Eigen::MatrixXd::Zero(f.is_cage_tv.sum(), 3);
    int j = 0;
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (f.is_cage_tv(i)) {
            max_grad_cage.row(j) = f.max_grad.row(i);
            max_grad_normalized_cage.row(j) = f.max_grad.row(i).normalized();
            j++;
        }
    }
    v_cage->addScalarQuantity("E_max", f.max_cage);
    v_cage->addVectorQuantity("\\nabla E_max", max_grad_cage);
    v_cage->addVectorQuantity("\\nabla E_max, normalized", max_grad_normalized_cage);

}