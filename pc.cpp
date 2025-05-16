#include "pc.h"

Eigen::MatrixXd base_cage_points = ico_pts;
Eigen::MatrixXi base_cage_faces = ico_faces;
Eigen::MatrixXi base_cage_tets = ico_tets;

// Eigen::MatrixXd base_cage_points = ico_pts_1;
// Eigen::MatrixXi base_cage_faces = ico_faces_1;
// Eigen::MatrixXi base_cage_tets = ico_tets_1;

// Eigen::MatrixXd base_cage_points = ico_pts_2;
// Eigen::MatrixXi base_cage_faces = ico_faces_2;
// Eigen::MatrixXi base_cage_tets = ico_tets_2;

bool is_close(double a, double b) {
    return fabs(a - b) < 1e-12;
}

template <typename T> std::vector<int> sort_indexes(const std::vector<T> &v) {

    // from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes

    // initialize original index locations
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values 
    std::stable_sort(idx.begin(), idx.end(),
        [&v](int i1, int i2) {return v[i1] < v[i2];});

    return idx;
}

template <typename T> std::vector<T> reorder_vector(const std::vector<T> &vals, const std::vector<int> &idxs) {
    std::vector<T> vals_new;
    for (int idx: idxs) {vals_new.push_back(vals[idx]);}
    return vals_new;
}

void setCageVerts(struct Faraday &f) {
    /*
        Updates f.is_cage_tv based on the cage vertices that the user has specified
        Will try to grow a ball of maximum size around around each user location
        Anything in the ball gets marked as a cage vertex
    */

    std::cout << "Updating cage verts." << std::endl;

    f.is_cage_tv = f.is_cage_tv_original;
    
    if (f.user_cages.size() > 0) {
        Eigen::MatrixXd Q_pos(f.user_cages.size(), 3);
        for (int i = 0; i < f.user_cages.size(); i++) {
            Q_pos.row(i) = f.TV.row(f.user_cages[i]);
        }
        Eigen::MatrixXi knn;
        igl::knn(Q_pos, f.P, 1, f.PI, f.CH, f.CN, f.W, knn);

        for (int i = 0; i < f.user_cages.size(); i++) {
            double radius = 0.9 * (f.P.row(knn(i, 0)) - Q_pos.row(i)).norm();
            Eigen::RowVectorXd off(3);
            off << radius, radius, radius;
            Eigen::VectorXd min_corner = Q_pos.row(i) - off;
            Eigen::VectorXd max_corner = Q_pos.row(i) + off;
            // filter for candidate points
            std::vector<int> candidates;
            for (int pt = 0; pt < f.TV.rows(); pt++) {
                if ( (f.TV.row(pt)(0) > min_corner(0)) && (f.TV.row(pt)(0) < max_corner(0)) &&
                     (f.TV.row(pt)(1) > min_corner(1)) && (f.TV.row(pt)(1) < max_corner(1)) &&
                     (f.TV.row(pt)(2) > min_corner(2)) && (f.TV.row(pt)(2) < max_corner(2)))
                candidates.push_back(pt);
            }
            for (int cd: candidates) {
                if ((f.TV.row(cd) - Q_pos.row(i)).norm() < radius) f.is_cage_tv(cd) = true;
            }
        }

    }
    

}

void addPointCharge(struct Faraday &f, int pt_idx) {
    if ((pt_idx >= 0) && (pt_idx < f.TV.rows()) && (!f.is_bdry_tv(pt_idx)) && (!f.is_cage_tv(pt_idx))) {
        auto it = std::find(f.point_charges.begin(), f.point_charges.end(), pt_idx);
        if (it == f.point_charges.end()) {
            f.point_charges.push_back(pt_idx);
        } else {
            std::cout << "You have already clicked here!" << std::endl;
        }
    } else if (pt_idx != -1) {
        std::cout << "Bad index for pt. constraint (may be a cage or boundary vtx.)" << std::endl;
    }
}

void addUserCage(struct Faraday &f, int pt_idx) {
    if ((pt_idx >= 0) && (pt_idx < f.TV.rows()) && (!f.is_bdry_tv(pt_idx)) && (!f.is_cage_tv(pt_idx))) {
        auto it = std::find(f.user_cages.begin(), f.user_cages.end(), pt_idx);
        if (it == f.user_cages.end()) {
            f.user_cages.push_back(pt_idx);
        } else {
            std::cout << "You have already clicked here!" << std::endl;
        }
    } else if (pt_idx != -1) {
        std::cout << "Bad index for pt. constraint (may be a cage or boundary vtx.)" << std::endl;
    }
}


int getQueryPointFromClicked(struct Faraday &f, int n, bool is_vertex) {

    Eigen::MatrixXd Q_pos;
    if (is_vertex) {
        Q_pos = f.SV.row(n);
    } else {
        Q_pos = ( f.SV.row(f.SF(n, 0)) + f.SV.row(f.SF(n, 1)) + f.SV.row(f.SF(n, 2)) ) / 3.;
    }

    int NUM_NEIGHS = 20;
    Eigen::MatrixXi res(1, NUM_NEIGHS);
    igl::knn(Q_pos, f.TV, NUM_NEIGHS, f.PI_TV, f.CH_TV, f.CN_TV, f.W_TV, res);

    Eigen::MatrixXd Q(NUM_NEIGHS, 3);
    Eigen::VectorXd res_wn;
    for (int i = 0; i < NUM_NEIGHS; i++) {
        Q.row(i) = f.TV.row(res(i));
    }

    igl::fast_winding_number(f.SV, f.SF, Q, res_wn);
    std::cout << res.row(0) << std::endl;
    std::cout << res_wn.transpose() << std::endl;

    int found = -1;
    int i = 0;
    for (int tv_idx: res.row(0)) {
        if (!(f.is_cage_tv(tv_idx)) && !(f.is_bdry_tv(tv_idx)) && (res_wn(i) < 0.05)) {found = tv_idx; break;}
        i++;
    }

    if (found == -1) std::cout << "Couldn't find a reasonable point in the vicinity of where you clicked..." << std::endl;

    return found;

}

void computeNearestNeighborDists(struct Faraday &f) {
    f.cage_radii = Eigen::VectorXd::Zero(f.P.rows());

    igl::parallel_for(f.P.rows(), [&](int i)
        {
            double r = 0.45 * (f.P.row(f.knn(i,1)) - f.P.row(i)).norm();
            f.cage_radii(i) = r;
        }
    , 100);
}


Eigen::MatrixXd stereographicProject(Eigen::VectorXd &pt, Eigen::MatrixXd &q, double alpha) {

    Eigen::MatrixXd q_stereo = Eigen::MatrixXd::Zero(q.rows(), q.cols());

    for (size_t i = 0; i < q.rows(); i++) {
        Eigen::VectorXd offset = (pt-q.row(i).transpose()) / (pt-q.row(i).transpose()).array().pow(2).sum();
        // std::cout << offset.transpose() << std::endl;
        q_stereo.row(i) = q.row(i).transpose() + alpha * offset; 
    }

    return q_stereo;

}

void findBdryCage(struct Faraday &f) {

    Eigen::VectorXi is_bdry_tv_new = Eigen::VectorXi::Zero(f.TV.rows());
    Eigen::VectorXi is_cage_tv_new = Eigen::VectorXi::Zero(f.TV.rows());

    for (size_t i = 0; i < f.TV.rows(); i++) {
        if ((i < f.is_cage_tv.rows()) && (i > 7)) {
            is_cage_tv_new[i] = f.is_cage_tv(i);
        } else if ( (is_close(f.TV(i,0), f.bb(0,0)) || is_close(f.TV(i,0), f.bb(1,0)) ) ||
                    (is_close(f.TV(i,1), f.bb(0,1)) || is_close(f.TV(i,1), f.bb(1,1)) )||
                    (is_close(f.TV(i,2), f.bb(0,2)) || is_close(f.TV(i,2), f.bb(1,2)) )
                    ) {
            f.TM(i) = 1;
            is_bdry_tv_new[i] = true;
        } 
    }

    f.is_cage_tv = is_cage_tv_new;
    f.is_cage_tv_original = is_cage_tv_new;
    f.is_bdry_tv = is_bdry_tv_new;

}

void fixBoundaryMarkers(struct Faraday &f) {
    /*
        If tetgen inserts a point on the edge of a boundary face,
        it marks it with "-1" instead of a marker associated with that face
        (probably because the edge is shared between two faces or whatever.)

        So now I have to go back and map each -1 to its appropriate boundary marker...
        The points inserted on the boundary (like, the box) get fixed in findBdryCage,
        so this is just for vertices inserted on the edges of the cage icosahedrons
    */

    std::vector<int> problematic;

    for (size_t i = 0; i < f.TV.rows(); i++) {
        if (f.TM(i) == -1) {
            problematic.push_back(i);
        }
    }

    Eigen::MatrixXd Q_problematic(problematic.size(), 3);

    size_t i = 0;
    for (int p: problematic) {
        Q_problematic.row(i) = f.TV.row(p);
        i++;
    }
    
    Eigen::MatrixXi knn_res(problematic.size(), 1);
    igl::knn(Q_problematic, f.P, 1, f.PI, f.CH, f.CN, f.W, knn_res);

    i = 0;
    for (int p: problematic) {
        f.TM(p) = knn_res(i, 0) + 2;
        i++;
    }

}

void findCageTets(struct Faraday &f) {
    std::vector<std::vector<int>> my_cage_tets(f.P.rows(), std::vector<int>());

    // Eigen::MatrixXi tet_vtx_ids(f.TT.rows(), 4);

    // for (size_t i = 0; i < f.TT.rows(); i++) {
    //     for (size_t j = 0; j < 4; j++) {
    //         tet_vtx_ids(i, j) = f.TM(f.TT(i,j));
    //     }
    // }

    // std::cout << tet_vtx_ids << std::endl;

    igl::parallel_for(f.TT.rows(), [&](int i) {
        Eigen::VectorXi votes = Eigen::VectorXi::Zero(f.P.rows()); // good lord

        for (int vtx: f.TT.row(i)) {
            if (f.TM(vtx) >= 2) {
                votes(f.TM(vtx) - 2) += 1;
            }
        }

        int most_votes = 0;
        int most_voted = -1;

        for (size_t s = 0; s < f.P.rows(); s++) {
            if (votes(s) > most_votes) {
                most_voted = s;
                most_votes = votes(s);
            }
        }

        if (most_votes == 3) {
            my_cage_tets[most_voted].push_back(i);
        }
    }, 100);

    f.my_cage_tets = my_cage_tets;
}

void findTets(struct Faraday &f) {

    std::vector<std::vector<int>> my_tets(f.TV.rows(), std::vector<int>());

    for (size_t i = 0; i < f.TT.rows(); i++) {
        for (int vtx: f.TT.row(i)) {
            my_tets[vtx].push_back(i);
        }
    }

    f.my_tets = my_tets;

}

void prepareTetgen(struct Faraday &f) {
    // ADD CAGE POINTS
    // Icosphere surrounding each interior point

    int START_CAGE = 8;

    std::vector<Eigen::Vector3d> add_rows_cage;
    Eigen::MatrixXd cage_vtx_mat(f.P.rows() * base_cage_points.rows(), 3);
    std::vector<Eigen::Vector3i> add_faces_cage;
    for (int i = 0; i < f.P.rows(); i++) {

        Eigen::Vector3d pt = f.P.row(i);

        size_t this_base = START_CAGE + add_rows_cage.size();

        for (int j = 0; j < base_cage_points.rows(); j++) {
            Eigen::Vector3d ico_pt = base_cage_points.row(j);
            Eigen::Vector3d new_pt = pt + ico_pt * f.cage_radii[i];
            add_rows_cage.push_back(new_pt);
            cage_vtx_mat.row(i * base_cage_points.rows() + j) = new_pt;
        }

        for (int j = 0; j < base_cage_faces.rows(); j++) {
            Eigen::Vector3i ico_face = base_cage_faces.row(j);
            add_faces_cage.push_back(Eigen::Vector3i::Ones() * this_base + ico_face);
        }

    }

    // pre-append corners of bounding box

    Eigen::MatrixXd BV;
    Eigen::MatrixXi BF;

    igl::bounding_box(cage_vtx_mat, BV, BF);

    double PADDING = 0.1;

    Eigen::Vector3d bb_max = BV.row(0);
    Eigen::Vector3d bb_min = BV.row(7);

    double pad = (bb_max - bb_min).cwiseAbs().minCoeff() * PADDING;

    igl::bounding_box(cage_vtx_mat, pad, BV, BF);
    
    Eigen::MatrixXd bb(2,3);
    bb.row(0) << BV.row(0);
    bb.row(1) << BV.row(7);

    std::vector<Eigen::Vector3d> add_rows_bdry;
    std::vector<Eigen::Vector3i> add_faces_bdry;

    // ADD BOUNDARY POINTS

    // corners
    add_rows_bdry.push_back(BV.row(0));
    add_rows_bdry.push_back(BV.row(1));
    add_rows_bdry.push_back(BV.row(2));
    add_rows_bdry.push_back(BV.row(3));
    add_rows_bdry.push_back(BV.row(4));
    add_rows_bdry.push_back(BV.row(5));
    add_rows_bdry.push_back(BV.row(6));
    add_rows_bdry.push_back(BV.row(7));
    add_faces_bdry.push_back(BF.row(0));
    add_faces_bdry.push_back(BF.row(1));
    add_faces_bdry.push_back(BF.row(2));
    add_faces_bdry.push_back(BF.row(3));
    add_faces_bdry.push_back(BF.row(4));
    add_faces_bdry.push_back(BF.row(5));
    add_faces_bdry.push_back(BF.row(6));
    add_faces_bdry.push_back(BF.row(7));
    add_faces_bdry.push_back(BF.row(8));
    add_faces_bdry.push_back(BF.row(9));
    add_faces_bdry.push_back(BF.row(10));
    add_faces_bdry.push_back(BF.row(11));
    
    // append points
    Eigen::MatrixXd V(add_rows_bdry.size() + add_rows_cage.size(), 3);
    Eigen::MatrixXd V_cage(add_rows_cage.size(), 3);
    Eigen::MatrixXi F(add_faces_bdry.size() + add_faces_cage.size(), 3);
    Eigen::VectorXi is_cage_tv(add_rows_bdry.size() + add_rows_cage.size());

    f.FM = Eigen::VectorXi::Zero(add_faces_bdry.size() + add_faces_cage.size());
    f.VM = Eigen::VectorXi::Zero(add_rows_bdry.size() + add_rows_cage.size());

    std::vector<std::vector<int>> my_cage_points(f.P.size(), std::vector<int>());
    
    size_t i = 0;

    for (Eigen::Vector3d row_to_add: add_rows_bdry) {
        V.row(i) = row_to_add;
        f.VM(i) = 1;
        i++;
    }

    size_t j = 0;
    for (Eigen::Vector3d row_to_add: add_rows_cage) {
        V.row(i) = row_to_add;
        V_cage.row(j) = row_to_add; 
        f.VM(i) = (j / base_cage_points.rows()) + 2;
        is_cage_tv[i] = 1;
        my_cage_points[(i - START_CAGE) / base_cage_points.rows()].push_back(i);
        i++;
        j++;
    }

    i = 0;

    for (Eigen::VectorXi face_to_add: add_faces_bdry) {
        F.row(i) = face_to_add;
        f.FM(i) = 1;
        i++;
    }
    
    j = 0;
    for (Eigen::VectorXi face_to_add: add_faces_cage) {
        F.row(i) = face_to_add;
        f.FM(i) = (j / base_cage_faces.rows()) + 2;
        i++;
        j++;
    }

    // set attributes of Faraday struct
    f.bb = bb;
    f.my_cage_points = my_cage_points;
    f.is_cage_tv = is_cage_tv;
    f.V = V;
    f.V_cage = V_cage;
    f.F = F;
    f.H = f.P;
    f.cage_faces = add_faces_cage;

}

void fillTetHoles(struct Faraday &f) {

    // also expands f.max to cover original points too

    f.TV_filled = f.TV;
    f.TT_filled = f.TT;
    f.max_filled = f.max;

    int base_num_tv = f.TV.rows();
    int base_num_tets = f.TT.rows();

    f.TV_filled.conservativeResize(base_num_tv + f.P.rows(), 3);
    f.TT_filled.conservativeResize(base_num_tets + (base_cage_faces.rows() * f.P.rows()), 4);
    f.max_filled.conservativeResize(base_num_tv + f.P.rows());

    for (size_t i = 0; i < f.P.rows(); i++) {

        f.TV_filled.row(base_num_tv + i) = f.P.row(i);
        f.max_filled(base_num_tv + i) = f.P_avg_value(i);

        // f.max_filled(base_num_tv + i) = 100.;

        for (size_t face = 0; face < base_cage_faces.rows(); face++) {

            Eigen::RowVectorXi this_tet(4);

            this_tet(0) = base_cage_tets.row(face)(0) == -1.
                            ? base_num_tv + i 
                            : base_cage_tets.row(face)(0) + ((i * base_cage_points.rows()) + 8);
            this_tet(1) = base_cage_tets.row(face)(1) == -1.
                            ? base_num_tv + i 
                            : base_cage_tets.row(face)(1) + ((i * base_cage_points.rows()) + 8);
            this_tet(2) = base_cage_tets.row(face)(2) == -1.
                            ? base_num_tv + i 
                            : base_cage_tets.row(face)(2) + ((i * base_cage_points.rows()) + 8);
            this_tet(3) = base_cage_tets.row(face)(3) == -1.
                            ? base_num_tv + i 
                            : base_cage_tets.row(face)(3) + ((i * base_cage_points.rows()) + 8);

            f.TT_filled.row(base_num_tets + ((i * base_cage_faces.rows()) + face)) = this_tet;
                
        }
    }

}

