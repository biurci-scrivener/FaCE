#include "solve.h"
#include <chrono>

Eigen::MatrixXd potential_dirs = my_view_directions;

Eigen::MatrixXd grad_tets(struct Faraday &f, Eigen::VectorXd &func) {
    Eigen::VectorXd res = f.grad * func;
    Eigen::MatrixXd g_f = Eigen::Map<Eigen::MatrixXd>(res.data(), f.TT.rows(), 3);
    return g_f;
}

Eigen::MatrixXd grad_tv(struct Faraday &f, Eigen::VectorXd &func) {
    Eigen::VectorXd res = f.grad * func;
    Eigen::MatrixXd g_f = Eigen::Map<Eigen::MatrixXd>(res.data(), f.TT.rows(), 3);
    return f.f_to_v * g_f;
}

void build_f_to_v_matrix(struct Faraday &f) {
    f.f_to_v = Eigen::SparseMatrix<double>(f.TV.rows(), f.TT.rows());
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < f.TV.rows(); i++) {
        double w_sum = 0;
        for (int tet: f.my_tets[i]) {
            w_sum += f.vols[tet];
        }
        for (int tet: f.my_tets[i]) {
            triplets.push_back(Eigen::Triplet<double>(i, tet, f.vols[tet] / w_sum));
        }
    }
    f.f_to_v.setFromTriplets(triplets.begin(), triplets.end());
}

Eigen::VectorXd barycentricTet(Eigen::MatrixXd &TV, const Eigen::VectorXi &tet, const Eigen::VectorXd &p) {
    Eigen::MatrixXd L(1, 4);
    igl::barycentric_coordinates(p.transpose(), TV.row(tet(0)), TV.row(tet(1)), TV.row(tet(2)), TV.row(tet(3)), L);
    return L.row(0);
}

// std::tuple<int, Eigen::VectorXd> findTetAndCoordinates(Eigen::MatrixXd &TV, Eigen::MatrixXi &TT, const Eigen::VectorXd &p) {

//     for (size_t i = 0; i < TT.rows(); i++) {
//         Eigen::VectorXd b = barycentricTet(TV, TT.row(i), p);
//         if (b.cwiseAbs().maxCoeff() > 1.00001) {
//             continue;
//         } else {
//             return std::make_tuple(i, b);
//         }
//     }

//     Eigen::VectorXd _none(4);
//     _none << 0.,0.,0.,0.;
//     return std::make_tuple(-1, _none);

// }

// std::unordered_map<int, std::tuple<int, Eigen::VectorXd>> computeRetargetingMap(struct Faraday &f_stereo, struct Faraday &f, Eigen::VectorXd &p) {

//     std::unordered_map<int, std::tuple<int, Eigen::VectorXd>> retarget_map;

//     Eigen::MatrixXd f_TV_stereo = stereographicProject(p, f.TV);

//     for (size_t i = 0; i < f.TV.rows(); i++) {
//         std::cout << "\t\tRetargeting " << i << std::endl;
//         std::tuple<int, Eigen::VectorXd> res = findTetAndCoordinates(f_stereo.TV, f_stereo.TT, f_TV_stereo.row(i));
//         if (std::get<0>(res) < 0) {
//             std::cout << "\tError retargeting: " << f_TV_stereo.row(i) << ", index " << i << std::endl;
//         } 
//         retarget_map.insert({i, res});
//     }

//     return retarget_map;
// }

// Eigen::VectorXd retargetFunction(struct Faraday &f_stereo, struct Faraday &f_to, std::unordered_map<int, std::tuple<int, Eigen::VectorXd>> &retarget_map, const Eigen::VectorXd &u) {
//     Eigen::VectorXd u_new = Eigen::VectorXd::Zero(u.rows());

//     for (size_t i = 0; i < u.rows(); i++) {
//         std::tuple<int, Eigen::VectorXd> bary = retarget_map[i];
//         int tet_idx = std::get<0>(bary);
//         if (tet_idx < 0) {
//             continue;
//         } else {
//             Eigen::VectorXi tet = f_stereo.TT.row(tet_idx);
//             Eigen::VectorXd bary_weights = std::get<1>(bary);
//             u_new(i) = bary_weights(0) * u(tet(0)) + bary_weights(1) * u(tet(1)) + bary_weights(2) * u(tet(2)) + bary_weights(3) * u(tet(3));
//         }
//     }

//     return u_new;
// }

// void remapStereoToOriginal(struct Faraday &f_stereo, struct Faraday &f, Eigen::VectorXd &p) {

//     std::cout << "\tComputing retargeting map" << std::endl;

//     std::unordered_map<int, std::tuple<int, Eigen::VectorXd>> retarget_map = computeRetargetingMap(f_stereo, f, p);

//     std::cout << "\tDone computing retargeting map" << std::endl;

//     for (int i = 0; i < f_stereo.u.cols(); i++) {
//         std::cout << "\tRemapping function " << i << std::endl;

//         Eigen::VectorXd res = retargetFunction(f_stereo, f, retarget_map, f_stereo.u.col(i));
//         f.u.col(i) = res;
//         Eigen::MatrixXd res_grad = grad_tv(f, res);
//         std::cout << "\t\tComputed grad. of u" << std::endl;
//         f.u_grad.col(i * 3) = res_grad.col(0);
//         f.u_grad.col(i * 3 + 1) = res_grad.col(1);
//         f.u_grad.col(i * 3 + 2) = res_grad.col(2);
//     }
// }


Eigen::VectorXd integrateMaxOverStereoPts(struct Faraday &f) {
    Eigen::VectorXd vals = Eigen::VectorXd::Zero(f.P.rows());
    igl::parallel_for(f.P.rows(), [&](int i) {
        double v = 0.;
        for (int cg: f.my_cage_points[i]) {v += f.max(cg);}
        vals(i) = v;}, 100);
    return vals;
}

void solvePotentialOverDirs(struct Faraday &f) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.5,1.5);
    
    std::cout << "Starting solve for potential over all directions." << std::endl;
    size_t no_fields = potential_dirs.rows();

    f.u = Eigen::MatrixXd::Zero(f.TV.rows(), no_fields);
    f.u_grad = Eigen::MatrixXd::Zero(f.TV.rows(), no_fields * 3);
    f.v_theta = Eigen::MatrixXd::Zero(f.TV.rows(), no_fields);
    f.v_theta_grad = Eigen::MatrixXd::Zero(f.TV.rows(), no_fields * 3);

    // pre-solve for Faraday effect, linear fields

    auto t1 = std::chrono::high_resolution_clock::now();

    Eigen::SparseMatrix<double> LHS = computeFaraday(f);

    std::cout << "Initializing solver, please wait." << std::endl;
    geometrycentral::PositiveDefiniteSolver<double> solver(LHS);

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::ratio<1,1>> dur_pre = t2 - t1;

    std::cout << "Initialized solver" << std::endl;
    
    /*

        Let u, v_theta be the potentials with and without shielding
        For each field direction, compute
            - u
            - v_theta
            - gradients of each (on vertices)

    */

    for (int i = 0; i < potential_dirs.rows(); i++) {
        std::cout << "\tSolving for direction " << i << std::endl;

        Eigen::VectorXd boundary_vals(f.TV.rows());
        Eigen::VectorXd dir = potential_dirs.row(i);
        std::cout << "\t" << dir.transpose() << std::endl;

        for (int j = 0; j < f.TV.rows(); j++) boundary_vals[j] = f.TV.row(j).dot(dir);

        f.v_theta.col(i) = boundary_vals;  
        std::cout << "\t\tComputed boundary values" << std::endl;

        // Eigen::MatrixXd boundary_vals_grad = grad_tv(f, boundary_vals);
        // std::cout << "\t\tComputed grad. of bdry. values" << std::endl;
        // f.v_theta_grad.col(i * 3) = boundary_vals_grad.col(0);
        // f.v_theta_grad.col(i * 3 + 1) = boundary_vals_grad.col(1);
        // f.v_theta_grad.col(i * 3 + 2) = boundary_vals_grad.col(2);

        Eigen::VectorXd res = solveFaraday(f, solver, boundary_vals);
        std::cout << "\t\tSolved for u" << std::endl;
        f.u.col(i) = res;
        Eigen::MatrixXd res_grad = grad_tv(f, res);
        std::cout << "\t\tComputed grad. of u" << std::endl;
        f.u_grad.col(i * 3) = res_grad.col(0);
        f.u_grad.col(i * 3 + 1) = res_grad.col(1);
        f.u_grad.col(i * 3 + 2) = res_grad.col(2);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1,1>> dur_solves = t3 - t2;

}

void solvePotentialPointCharges(struct Faraday &f, std::vector<int> &pt_constraints, double charge_scale) {

    if (!pt_constraints.size()) {
        f.u.conservativeResize(f.u.rows(), potential_dirs.rows());
        f.u_grad.conservativeResize(f.u_grad.rows(), (potential_dirs.rows() * 3));
        f.v_theta.conservativeResize(f.v_theta.rows(), potential_dirs.rows());
        f.v_theta_grad.conservativeResize(f.v_theta_grad.rows(), (potential_dirs.rows() * 3));
        std::cout << "\tNo point constaints specified, returning." << std::endl;
        return;
    }

    std::cout << "\tSolving for point charge field" << std::endl;

    f.u.conservativeResize(f.u.rows(), potential_dirs.rows() + 1);
    f.u_grad.conservativeResize(f.u_grad.rows(), (potential_dirs.rows() * 3) + 3);
    f.v_theta.conservativeResize(f.v_theta.rows(), potential_dirs.rows() + 1);
    f.v_theta_grad.conservativeResize(f.v_theta_grad.rows(), (potential_dirs.rows() * 3) + 3);

    // pre-solve for Faraday effect, point charges

    std::unordered_map<int, int> global_to_matrix_ordering_pt_charge;
    Eigen::SparseMatrix<double> LHS;
    std::tie(global_to_matrix_ordering_pt_charge, LHS) = computeFaraday(f, pt_constraints);

    std::cout << "Initializing solver, please wait." << std::endl;
    geometrycentral::PositiveDefiniteSolver<double> solver_pt_charge(LHS);
    std::cout << "Initialized solver" << std::endl;

    // solve

    int i = potential_dirs.rows();

    double pt_const_val = charge_scale * f.v_theta.col(0).maxCoeff();
    Eigen::VectorXd pt_base = Eigen::VectorXd::Zero(f.TV.rows());
    f.v_theta.col(i) = pt_base;

    Eigen::MatrixXd boundary_vals_grad = grad_tv(f, pt_base);
    f.v_theta_grad.col(i * 3) = boundary_vals_grad.col(0);
    f.v_theta_grad.col(i * 3 + 1) = boundary_vals_grad.col(1);
    f.v_theta_grad.col(i * 3 + 2) = boundary_vals_grad.col(2);

    Eigen::VectorXd res = solveFaraday(f, solver_pt_charge, global_to_matrix_ordering_pt_charge, pt_constraints, pt_const_val);
    f.u.col(i) = res;
    Eigen::MatrixXd res_grad = grad_tv(f, res);
    f.u_grad.col(i * 3) = res_grad.col(0);
    f.u_grad.col(i * 3 + 1) = res_grad.col(1);
    f.u_grad.col(i * 3 + 2) = res_grad.col(2);
    
}

void solveMaxFunction(struct Faraday &f) {

    /*

        Let u be the potential with shielding (for a given direction)
        For each field direction, compute |grad_u|
            (Note: these gradients live on vertices)

        Then, take the maximum value (of this norm) across all directions
        Finally, take gradient (on vertices)

    */

    std::cout << "Computing max. grad." << std::endl;

    f.gradmag = Eigen::MatrixXd::Zero(f.TV.rows(), f.u.cols());
    f.max = Eigen::VectorXd::Zero(f.TV.rows());
    f.max_var = Eigen::VectorXd::Zero(f.TV.rows());
    f.max_grad = Eigen::MatrixXd::Zero(f.TV.rows(), 3);

    for (int i = 0; i < f.u.cols(); i++) {
        Eigen::VectorXd grad_norm = (f.u_grad.middleCols(i * 3, 3)).rowwise().norm();
        f.gradmag.col(i) = grad_norm;
    }

    // sort each row of gradmag
    f.gradmag_sorted = Eigen::MatrixXd::Zero(f.TV.rows(), f.u.cols());
    for (int i = 0; i < f.gradmag.rows(); i++) {
        Eigen::RowVectorXd this_row = f.gradmag.row(i);
        std::sort(this_row.data(), this_row.data() + this_row.size());
        f.gradmag_sorted.row(i) = this_row;
    }

    f.max = f.gradmag.rowwise().maxCoeff();
    // compute variance
    Eigen::VectorXd max_mean = f.gradmag.rowwise().mean();
    f.max_var = Eigen::VectorXd::Zero(f.TV.rows());
    for (int i = 0; i < f.gradmag.rows(); i++) {
        f.max_var(i) = sqrt((f.gradmag.row(i).transpose() - Eigen::VectorXd::Constant(f.gradmag.cols(), max_mean(i))).array().pow(2).sum()) / ((double)f.gradmag.cols());
    }

    f.max_grad_tets = grad_tets(f, f.max);
    f.max_grad = f.f_to_v * f.max_grad_tets;
    f.max_grad_normalized = f.max_grad.rowwise().normalized();
    f.max_grad_tets_normalized = f.max_grad_tets.rowwise().normalized();
    f.max_grad_norm = f.max_grad.rowwise().norm();
    

}

void estimateNormals(struct Faraday &f) {

    std::cout << "Estimating normals" << std::endl;
    f.N_est = Eigen::MatrixXd::Zero(f.P.rows(), 3);
    f.P_max_grad_norm = Eigen::VectorXd::Zero(f.P.rows());
    
    /*
    
        Normals are estimated from max_grad

    */

    /*
        Estimate from tet. vertices
    */

    for (int i = 0; i < f.P.rows(); i++) {
        for (int cage: f.my_cage_points[i]) {
            f.N_est.row(i) += f.max_grad.row(cage);
        }
        f.N_est.row(i) /= f.my_cage_points[i].size();
        f.P_max_grad_norm(i) = f.N_est.row(i).norm();
        f.N_est.row(i) = f.N_est.row(i).normalized();
    }

    /*
        Estimate from tets.
    */
    
    // for (int i = 0; i < f.P.rows(); i++) {
    //     double weight_sum = 0;
    //     for (int cage_tet: f.my_cage_tets[i]) {
    //         f.N_est.row(i) += f.max_grad_tets.row(cage_tet) * f.vols(cage_tet);
    //         weight_sum += f.vols(cage_tet);
    //     }
    //     f.N_est.row(i) /= weight_sum;
    //     f.N_est.row(i) = f.N_est.row(i).normalized();
    // }

}

void computePointsetFunctions(struct Faraday &f) {
    // puts interesting quantities on the input points
    // based on f.max

    f.P_max = Eigen::VectorXd::Zero(f.P.rows());
    f.P_avg_value = Eigen::VectorXd::Zero(f.P.rows());
    f.P_var = Eigen::VectorXd::Zero(f.P.rows());
    f.max_cage = Eigen::VectorXd::Zero(f.is_cage_tv.sum());

    f.is_interior = Eigen::VectorXd::Zero(f.P.rows());

    size_t j = 0;
    for (int i = 0; i < f.P.rows(); i++) {
        double cage_sum = 0.;
        double max_cur = 0.;
        for (int cage: f.my_cage_points[i]) {
            f.max_cage(j) = f.max(cage); // this is the usual max function, just on the cage vertices
            cage_sum += f.max(cage);
            max_cur = f.max(cage) > max_cur ? f.max(cage) : max_cur;
            j++;
        }
        f.P_avg_value(i) = cage_sum / f.my_cage_points[i].size();
        f.P_max(i) = max_cur;
        double var = 0.;
        for (int cage: f.my_cage_points[i]) {
            var += pow(f.P_avg_value(i) - f.max(cage), 2);
        }
        var /= f.my_cage_points[i].size();
        var = sqrt(var);
        f.P_var(i) = var;
    }

}

void classifyInterior(struct Faraday &f, double iso) {

    
    
    std::cout << "Classifying interior points" << std::endl;
    std::cout << "\tSurfacing with isovalue " << iso << std::endl;

    Eigen::MatrixXd SV;
    Eigen::MatrixXi SF;

    igl::marching_tets(f.TV_filled, f.TT_filled, f.max_filled, iso, SV, SF);

    std::cout << "\tRunning fast winding number (f.P)" << std::endl;
    igl::fast_winding_number(SV, SF, f.P, f.wn);
    
    for (int i = 0; i < f.P.rows(); i++) {
        f.is_interior(i) = f.wn(i) > 0.9;
    }

    // std::vector<int> sorted_indices(f.P.rows());
    // for (int i = 0; i < sorted_indices.size(); i++) {
    //     sorted_indices[i] = i;
    // }
    
    // std::sort(sorted_indices.data(), sorted_indices.data() + sorted_indices.size(), [&f](int i1, int i2){return f.P_max[i1] < f.P_max[i2];});

    // int in_streak = 0;
    // std::cout << "\tInitializing aabb tree" << std::endl;
    // f.aabb_tree.init(f.SV, f.SF);
    // std::cout << "\tDone" << std::endl;

    // int streak = 0;
    // for (int i = 0; i < sorted_indices.size(); i++) {

    //     if (i % 100 == 0) std::cout << "\tProcessing " << i << std::endl;

    //     bool in = f.is_interior(sorted_indices[i]) = in_out_by_rayshooting(f, sorted_indices[i]);
    //     if (in) {
    //         f.is_interior(sorted_indices[i]) = 1.;
    //         streak = 0;
    //     } else {
    //         streak++;
    //     }

    //     if (streak > 5) {
    //         std::cout << "\tDone, hit streak. Total processed: " << i + 1 << std::endl;
    //         break;
    //     }
    // }

    // for (int i = 0; i < f.P.rows(); i++) {
    //     f.is_interior(i) = f.P_var(i) < thresh;
    // }

}

// bool in_out_by_rayshooting(struct Faraday &f, int pt_idx) {
//     size_t count_in = 0;
//     for (int j = 0; j < ico_pts_2.rows(); j++) {
//         std::vector<igl::Hit> hits;
//         f.aabb_tree.intersect_ray(f.SV, f.SF, f.P.row(pt_idx), ico_pts_2.row(j), hits);
//         count_in += hits.size() % 2 == 1;
//     }

//     return (((double)count_in) / ico_pts_2.size() > 0.75);
// }

Eigen::VectorXd solveFaraday(struct Faraday &f, geometrycentral::PositiveDefiniteSolver<double> &solver, Eigen::VectorXd &bdry_vals) {

    int boundary_count = f.is_bdry_tv.sum();
    int cage_count = f.is_cage_tv.sum();
    int interior_count = f.TV.rows() - cage_count - boundary_count;

    // build RHS
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(interior_count + 1);
    std::cout << "\t\tBuilding RHS" << std::endl;

    for (int k=0; k<f.L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(f.L,k); it; ++it) {

            // this is just so that I can do an "is_interior" type call
            // using the interior_count thing from computeFaraday
            int i = f.global_to_matrix_ordering[it.row()];

            if ((i < interior_count) && (f.is_bdry_tv(it.col()))) {
                RHS[i] -= it.value() * bdry_vals(it.col());
            }

        }
    }

    std::cout << "\t\tSolving" << std::endl;
    Eigen::VectorXd u = solver.solve(RHS);
    std::cout << "\t\tSolved" << std::endl;

    Eigen::VectorXd sol = Eigen::VectorXd::Zero(f.TV.rows());
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (f.is_bdry_tv(i)) {
            sol[i] = bdry_vals[i];
        } else if (f.is_cage_tv(i)) {
            sol[i] = u[interior_count];
        } else {
            sol[i] = u[f.global_to_matrix_ordering[i]];
        }
    }

    return sol;
}

Eigen::SparseMatrix<double> computeFaraday(struct Faraday &f) {

    std::unordered_map<int, int> global_to_matrix_ordering;
    int matrix_count = 0;
    int boundary_count = f.is_bdry_tv.sum();
    int cage_count = f.is_cage_tv.sum();
    int interior_count = f.TV.rows() - cage_count - boundary_count;

    for (size_t i = 0; i < f.TV.rows(); i++) {
        // "normal" interior vertices
        if ((!f.is_bdry_tv(i)) && (!f.is_cage_tv(i))) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    for (size_t i = 0; i < f.TV.rows(); i++) {
        // boundary vertices
        if (f.is_bdry_tv(i)) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    for (size_t i = 0; i < f.TV.rows(); i++) {
        // cage vertices
        if (f.is_cage_tv(i)) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    
    if (matrix_count != f.TV.rows()) {throw std::runtime_error("Not every cell was indexed");}

    std::cout << "\tDone reindexing" << std::endl;

    Eigen::SparseMatrix<double> LHS(interior_count + 1, interior_count + 1);
    std::vector<Eigen::Triplet<double>> triplets;

    // split Laplacian into parts
    Eigen::SparseMatrix<double> L_NC(interior_count, interior_count);
    Eigen::SparseMatrix<double> L_CC(cage_count, cage_count);
    Eigen::SparseMatrix<double> L_NCC(interior_count, cage_count);
    Eigen::SparseMatrix<double> L_CNC(cage_count, interior_count);
    // std::vector<Eigen::Triplet<double>> L_NC_triplets;
    std::vector<Eigen::Triplet<double>> L_CC_triplets;
    std::vector<Eigen::Triplet<double>> L_NCC_triplets;
    std::vector<Eigen::Triplet<double>> L_CNC_triplets;

    std::cout << "Setting triplets" << std::endl;

    int offset = interior_count + boundary_count;
    for (int k=0; k<f.L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(f.L,k); it; ++it) {
            int i = global_to_matrix_ordering[it.row()];
            int j = global_to_matrix_ordering[it.col()];
            if ((i < interior_count) && (f.is_cage_tv(it.col()))) {
                L_NCC_triplets.push_back(Eigen::Triplet<double>(i, j - offset, it.value()));
            } else if ((j < interior_count) && (f.is_cage_tv(it.row()))) {
                L_CNC_triplets.push_back(Eigen::Triplet<double>(i - offset, j, it.value()));
            } else if ((i < interior_count) && (j < interior_count)) {
                triplets.push_back(Eigen::Triplet<double>(i, j, it.value()));
            } else if ((f.is_cage_tv(it.row())) && (f.is_cage_tv(it.col()))) {
                L_CC_triplets.push_back(Eigen::Triplet<double>(i - offset, j - offset, it.value()));
            }
        }
    }

    L_CC.setFromTriplets(L_CC_triplets.begin(), L_CC_triplets.end());
    L_NCC.setFromTriplets(L_NCC_triplets.begin(), L_NCC_triplets.end());
    L_CNC.setFromTriplets(L_CNC_triplets.begin(), L_CNC_triplets.end());

    Eigen::VectorXd col_r = L_NCC * Eigen::VectorXd::Ones(cage_count);
    Eigen::VectorXd row_b = Eigen::RowVectorXd::Ones(cage_count) * L_CNC;
    double val_br = Eigen::RowVectorXd::Ones(cage_count) * L_CC * Eigen::VectorXd::Ones(cage_count);

    for (int i = 0; i < interior_count; i++) {
        triplets.push_back(Eigen::Triplet<double>(interior_count, i, row_b(i)));
        triplets.push_back(Eigen::Triplet<double>(i, interior_count, col_r(i)));
    }
    triplets.push_back(Eigen::Triplet<double>(interior_count, interior_count, val_br));

    LHS.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << "\tDone" << std::endl;

    f.global_to_matrix_ordering = global_to_matrix_ordering;
    return LHS;

}

Eigen::VectorXd solveFaraday(struct Faraday &f, geometrycentral::PositiveDefiniteSolver<double> &solver, std::unordered_map<int, int> &global_to_matrix_ordering, std::vector<int> &pt_constraints, double const_val) {

    // version for point constraints

    int boundary_count = f.is_bdry_tv.sum() + pt_constraints.size();
    int cage_count = f.is_cage_tv.sum();
    int interior_count = f.TV.rows() - cage_count - boundary_count;

    // build RHS
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(interior_count + 1);
    std::cout << "\t\tBuilding RHS" << std::endl;

    for (int k=0; k<f.L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(f.L,k); it; ++it) {

            // this is just so that I can do an "is_interior" type call
            // using the interior_count thing from computeFaraday
            int i = global_to_matrix_ordering[it.row()];
            
            if ((i < interior_count) && (std::find(pt_constraints.begin(), pt_constraints.end(), it.col()) != pt_constraints.end())) {
                RHS[i] -= it.value() * const_val;
            }

        }
    }

    std::cout << "\tSolving" << std::endl;
    Eigen::VectorXd u = solver.solve(RHS);
    std::cout << "\tSolved" << std::endl;

    // build solution vector

    Eigen::VectorXd sol = Eigen::VectorXd::Zero(f.TV.rows());
    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (std::find(pt_constraints.begin(), pt_constraints.end(), i) != pt_constraints.end()) {
            sol[i] = const_val;
        } else if (f.is_cage_tv(i)) {
            sol[i] = u[interior_count];
        } else if (!f.is_bdry_tv(i)){
            sol[i] = u[global_to_matrix_ordering[i]];
        }
    }

    std::cout << "\tBuilt solution vector" << std::endl;

    return sol;
    
}

std::tuple<std::unordered_map<int, int>, Eigen::SparseMatrix<double>> computeFaraday(struct Faraday &f, std::vector<int> &pt_constraints) {

    // version for point constraints

    std::unordered_map<int, int> global_to_matrix_ordering;
    int matrix_count = 0;
    int boundary_count = f.is_bdry_tv.sum() + pt_constraints.size();
    int cage_count = f.is_cage_tv.sum();
    int interior_count = f.TV.rows() - cage_count - boundary_count;

    for (size_t i = 0; i < f.TV.rows(); i++) {
        // "normal" interior vertices
        if ((!f.is_bdry_tv(i)) && (!f.is_cage_tv(i)) && (std::find(pt_constraints.begin(), pt_constraints.end(), i) == pt_constraints.end())) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    for (size_t i = 0; i < f.TV.rows(); i++) {
        // boundary vertices
        if (f.is_bdry_tv(i) || (std::find(pt_constraints.begin(), pt_constraints.end(), i) != pt_constraints.end())) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    for (size_t i = 0; i < f.TV.rows(); i++) {
        // cage vertices
        if (f.is_cage_tv(i)) {
            global_to_matrix_ordering.insert({i, matrix_count});
            matrix_count++;
        }
    }
    
    if (matrix_count != f.TV.rows()) {throw std::runtime_error("Not every cell was indexed");}

    std::cout << "\tDone reindexing" << std::endl;

    Eigen::SparseMatrix<double> LHS(interior_count + 1, interior_count + 1);
    std::vector<Eigen::Triplet<double>> triplets;

    // split Laplacian into parts
    Eigen::SparseMatrix<double> L_NC(interior_count, interior_count);
    Eigen::SparseMatrix<double> L_CC(cage_count, cage_count);
    Eigen::SparseMatrix<double> L_NCC(interior_count, cage_count);
    Eigen::SparseMatrix<double> L_CNC(cage_count, interior_count);
    // std::vector<Eigen::Triplet<double>> L_NC_triplets;
    std::vector<Eigen::Triplet<double>> L_CC_triplets;
    std::vector<Eigen::Triplet<double>> L_NCC_triplets;
    std::vector<Eigen::Triplet<double>> L_CNC_triplets;

    std::cout << "Setting triplets" << std::endl;

    int offset = interior_count + boundary_count;
    for (int k=0; k<f.L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(f.L,k); it; ++it) {
            int i = global_to_matrix_ordering[it.row()];
            int j = global_to_matrix_ordering[it.col()];
            if ((i < interior_count) && (f.is_cage_tv(it.col()))) {
                L_NCC_triplets.push_back(Eigen::Triplet<double>(i, j - offset, it.value()));
            } else if ((j < interior_count) && (f.is_cage_tv(it.row()))) {
                L_CNC_triplets.push_back(Eigen::Triplet<double>(i - offset, j, it.value()));
            } else if ((i < interior_count) && (j < interior_count)) {
                triplets.push_back(Eigen::Triplet<double>(i, j, it.value()));
            } else if ((f.is_cage_tv(it.row())) && (f.is_cage_tv(it.col()))) {
                L_CC_triplets.push_back(Eigen::Triplet<double>(i - offset, j - offset, it.value()));
            }
        }
    }

    L_CC.setFromTriplets(L_CC_triplets.begin(), L_CC_triplets.end());
    L_NCC.setFromTriplets(L_NCC_triplets.begin(), L_NCC_triplets.end());
    L_CNC.setFromTriplets(L_CNC_triplets.begin(), L_CNC_triplets.end());

    Eigen::VectorXd col_r = L_NCC * Eigen::VectorXd::Ones(cage_count);
    Eigen::VectorXd row_b = Eigen::RowVectorXd::Ones(cage_count) * L_CNC;
    double val_br = Eigen::RowVectorXd::Ones(cage_count) * L_CC * Eigen::VectorXd::Ones(cage_count);

    for (int i = 0; i < interior_count; i++) {
        triplets.push_back(Eigen::Triplet<double>(interior_count, i, row_b(i)));
        triplets.push_back(Eigen::Triplet<double>(i, interior_count, col_r(i)));
    }
    triplets.push_back(Eigen::Triplet<double>(interior_count, interior_count, val_br));

    LHS.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << "\tDone" << std::endl;

    return std::make_tuple(global_to_matrix_ordering, LHS);

}

void extract_isosurface(struct Faraday &f, double isoval) {
    std::cout << "Surfacing with value " << isoval << std::endl;
    // f.max.minCoeff() + 0.1 * (f.max.array().mean() - f.max.minCoeff())
    // igl::marching_tets(f.TV, f.TT, f.max, isoval, f.SV, f.SF);
    igl::marching_tets(f.TV_filled, f.TT_filled, f.max_filled, isoval, f.SV, f.SF);
}

void faster_igl_quantities(struct Faraday &f) {

    /*
        Hopefully, identical to the libigl versions but with parallel_for
    */

    const int n = f.TV.rows(); int m = f.TT.rows();

    /*
        F = [ ...
        T(:,1) T(:,2) T(:,3); ...
        T(:,1) T(:,3) T(:,4); ...
        T(:,1) T(:,4) T(:,2); ...
        T(:,2) T(:,4) T(:,3)]; */
    Eigen::MatrixXi F(4*m,3);
    igl::parallel_for(m, [&](int i) {
        F.row(0*m + i) << f.TT(i,0), f.TT(i,1), f.TT(i,2);
        F.row(1*m + i) << f.TT(i,0), f.TT(i,2), f.TT(i,3);
        F.row(2*m + i) << f.TT(i,0), f.TT(i,3), f.TT(i,1);
        F.row(3*m + i) << f.TT(i,1), f.TT(i,3), f.TT(i,2);
    }, 100);

    // compute volume of each tet
    // igl::volume(f.TV,f.TT,f.vols);

    // std::cout << "Here 1" << std::endl;

    f.vols.resize(m,1);
    igl::parallel_for(m, [&](int t) {
        const Eigen::RowVector3d & a = f.TV.row(f.TT(t,0));
        const Eigen::RowVector3d & b = f.TV.row(f.TT(t,1));
        const Eigen::RowVector3d & c = f.TV.row(f.TT(t,2));
        const Eigen::RowVector3d & d = f.TV.row(f.TT(t,3));
        f.vols(t) = -(a-d).dot((b-d).cross(c-d))/6.;
    }, 100);

    // std::cout << "Here 2" << std::endl;

    Eigen::VectorXd A = Eigen::VectorXd::Zero(F.rows());
    Eigen::MatrixXd N(F.rows(),3);

    // compute tetrahedron face normals
    igl::per_face_normals(f.TV,F,N); int norm_rows = N.rows();

    // std::cout << "Here 2.5" << std::endl;

    // shouldn't these already be normalized??? 
    igl::parallel_for(norm_rows, [&](int i) {
         N.row(i) /= N.row(i).norm();
    }, 100);
       
    // std::cout << "Here 2.75" << std::endl;

    // igl::doublearea(f.TV,F,A); A/=2.;

    const size_t n_faces = F.rows();
    // Projected area helper
    const auto & proj_doublearea =
        [&f, &F](const int x, const int y, const int f_idx)
    {
        auto rx = f.TV(F(f_idx,0),x)-f.TV(F(f_idx,2),x);
        auto sx = f.TV(F(f_idx,1),x)-f.TV(F(f_idx,2),x);
        auto ry = f.TV(F(f_idx,0),y)-f.TV(F(f_idx,2),y);
        auto sy = f.TV(F(f_idx,1),y)-f.TV(F(f_idx,2),y);
        return rx*sy - ry*sx;
    };

    igl::parallel_for(n_faces, [&](int f_idx)
    {
        for(int d = 0;d<3;d++)
        {
            const auto dblAd = proj_doublearea(d,(d+1)%3,f_idx);
            A(f_idx) += dblAd*dblAd;
        }
    }, 100);
    

    A = A.array().sqrt().eval();

    // std::cout << "Here 3" << std::endl;

    std::vector<Eigen::Triplet<double> > G_t;

    for (int i = 0; i < 4 * m; i++) {
        int T_j; // j indexes : repmat([T(:,4);T(:,2);T(:,3);T(:,1)],3,1)
        switch (i/m) {
            case 0:
            T_j = 3;
            break;
            case 1:
            T_j = 1;
            break;
            case 2:
            T_j = 2;
            break;
            case 3:
            T_j = 0;
            break;
        }
        int i_idx = i%m;
        int j_idx = f.TT(i_idx,T_j);

        double val_before_n = A(i)/(3*f.vols(i_idx));
        G_t.push_back(Eigen::Triplet<double>(0*m+i_idx, j_idx, val_before_n * N(i,0)));
        G_t.push_back(Eigen::Triplet<double>(1*m+i_idx, j_idx, val_before_n * N(i,1)));
        G_t.push_back(Eigen::Triplet<double>(2*m+i_idx, j_idx, val_before_n * N(i,2)));
    }

    // std::cout << "Here 4" << std::endl;
    f.grad.resize(3*m,n);
    f.grad.setFromTriplets(G_t.begin(), G_t.end());
}