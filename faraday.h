#ifndef FARADAY_STRUCT
#define FARADAY_STRUCT

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <igl/AABB.h>

struct Faraday {

    /*
        For each member of the struct, I note which function is responsible for initializing it
    */

    Eigen::MatrixXd P; // main()
    Eigen::MatrixXd N; // main()
    /*
        The 3D positions of the cage vertices: just used to create a point cloud in Polyscope
        Otherwise, these are just part of the tet mesh verts. (f.TV)
    */
    Eigen::MatrixXd V_cage; // prepareTetgen()
    /*
        For each point in P, stores the indices of its associated cage vertices
    */
	std::vector<std::vector<int>> my_cage_points; // prepareTetgen()
	Eigen::MatrixXd bb; // prepareTetgen()

    // ========================================================================
    // tet stuff

    Eigen::MatrixXd V; // prepareTetgen()
    Eigen::MatrixXi F; // prepareTetgen()
    Eigen::MatrixXd H; // prepareTetgen()
    Eigen::VectorXi VM,FM; // dummy inputs, unused
    Eigen::MatrixXd R; // dummy inputs, unused
    Eigen::MatrixXd TV; // igl::copyleft::tetgen::tetrahedralize()
	Eigen::MatrixXi TT; // igl::copyleft::tetgen::tetrahedralize()
	Eigen::MatrixXi TF; // igl::copyleft::tetgen::tetrahedralize()
    Eigen::VectorXi TM,TR,PT; // igl::copyleft::tetgen::tetrahedralize(), unused
    Eigen::MatrixXi FT,TN; // igl::copyleft::tetgen::tetrahedralize(), unused
    Eigen::MatrixXd BC; // igl::copyleft::tetgen::tetrahedralize()
    int num_regions; // igl::copyleft::tetgen::tetrahedralize(), unused

    // ========================================================================
    // octree & knn stuff

    std::vector<std::vector<int>> PI; // igl::octree()
    Eigen::MatrixXi CH; // igl::octree()
    Eigen::MatrixXd CN; // igl::octree()
    Eigen::VectorXd W; // igl::octree()
    Eigen::MatrixXi knn; // igl::knn()

    Eigen::VectorXd cage_radii; // computeNearestNeighborDists()

    Eigen::VectorXi is_cage_tv; // findBdryCage()
    Eigen::VectorXi is_cage_tv_original; // findBdryCage()
    Eigen::VectorXi is_bdry_tv; // findBdryCage()

    // ========================================================================
    // another octree... but for the tet mesh

    std::vector<std::vector<int>> PI_TV; // igl::octree()
    Eigen::MatrixXi CH_TV; // igl::octree()
    Eigen::MatrixXd CN_TV; // igl::octree()
    Eigen::VectorXd W_TV; // igl::octree()

    // ========================================================================

    /*
        For each point in TV, stores the indices of associated tets
    */
    std::vector<std::vector<int>> my_tets; // findTets()
    std::vector<std::vector<int>> my_cage_tets; // findCageTets()

    // ========================================================================
    // numerical

    Eigen::SparseMatrix<double> M; // igl::cotmatrix()
    Eigen::SparseMatrix<double> D_inv; // unused, not set anymore
    Eigen::SparseMatrix<double> L; // runPipeline(): is either M or M * M
    Eigen::SparseMatrix<double> grad; // faster_igl_quantities()
    Eigen::VectorXd vols; // faster_igl_quantities()

    // ========================================================================
    // solver

    std::unordered_map<int, int> global_to_matrix_ordering; // computeFaraday()

    // ========================================================================

    /*
        Turns a function on faces (tets.) into a function on verts.
    */
    Eigen::SparseMatrix<double> f_to_v; // build_f_to_v_matrix()

    // ========================================================================
    // Faraday solve

    /*
        Each column corresponds to a linear field
        Each row corresponds to a tet. vertex
            and contains the potentials for that vertex (all fields)
    */
    Eigen::MatrixXd u; // solvePotentialOverDirs(), solvePotentialPointCharges()
    /*
        Every 3 columns correspond to a linear field (3 entries for gradient)
        Each row corresponds to a tet. vertex
            and contains the gradients of the potentials for that vertex (all fields)
    */
    Eigen::MatrixXd u_grad; // solvePotentialOverDirs(), solvePotentialPointCharges()
    Eigen::MatrixXd v_theta; // unset, unused
    Eigen::MatrixXd v_theta_grad; // unset, unused
    
    /*
        Each column corresponds to a linear field
        Each row corresponds to a tet. vertex
            and contains the magnitudes of the gradients of the potentials for that vertex (all fields)
    */
    Eigen::MatrixXd gradmag; // solveMaxFunction()
    /*
        Like gradmag, but each row is sorted in ascending order. Used by the "view rank" button
    */
    Eigen::MatrixXd gradmag_sorted; // solveMaxFunction()
    /*
        The super duper special function that we use for just about everything
        AKA "Max. mag. grad."
        Corresponds to max_{norm(grad(u_theta))} over all directions theta, for each vertex 
    */
    Eigen::VectorXd max; // solveMaxFunction()
    /*
        Gradients of the super duper special function (f.max) on tets
    */
    Eigen::MatrixXd max_grad_tets; // solveMaxFunction()
    /*
        Gradients of the super duper special function (f.max) on tet. vertices
    */
    Eigen::MatrixXd max_grad; // solveMaxFunction()
    /*
        Identical to f.max but sized for the cage vertices (for Polyscope)
    */
    Eigen::VectorXd max_cage; // computePointsetFunctions()
    /*
        Corresponds to variance( norm(grad(u_theta)) ) over all directions theta, for each vertex 
        There is no sense in which this is a "max" and it should be renamed...
    */
    Eigen::VectorXd max_var; // solveMaxFunction()
    /*
        Gradient of f.max, but normalized
    */
    Eigen::MatrixXd max_grad_normalized; // solveMaxFunction()
    Eigen::MatrixXd max_grad_tets_normalized; // solveMaxFunction()
    /*
        This is just f.max averaged onto the vertices in P 
            from their associated cage vertices.
        THe name is confusing, sorry
    */
    Eigen::VectorXd P_avg_value; // computePointsetFunctions()
    /*
        Norm of max_grad on points
    */
    Eigen::VectorXd P_max_grad_norm;
    /*
        This is the variance of f.max over all the cage vertices
            associated with each point in P
    */
    Eigen::VectorXd P_var; // computePointsetFunctions()
    /*
        This is the MAX of f.max over all the cage vertices 
            associated with each point in P
    */
    Eigen::VectorXd P_max; // computePointsetFunctions()

    Eigen::VectorXd max_grad_norm;

    // ========================================================================
    // Estimates, scoring 

    Eigen::MatrixXd N_est; // estimateNormals()
    Eigen::VectorXd is_interior; // classifyInterior()

    // ========================================================================

    // Isosurace extraction
    std::vector<Eigen::Vector3i> cage_faces;
    Eigen::VectorXd max_filled;
    Eigen::MatrixXd TV_filled;
    Eigen::MatrixXi TT_filled;
    Eigen::MatrixXd SV;
    Eigen::MatrixXi SF;
    igl::AABB<Eigen::MatrixXd, 3> aabb_tree;
    Eigen::VectorXd wn;

    // ========================================================================

    std::vector<int> point_charges;
    std::vector<int> user_cages;

    // ========================================================================

    Eigen::MatrixXd AV;
    Eigen::MatrixXi AF;

};

#endif