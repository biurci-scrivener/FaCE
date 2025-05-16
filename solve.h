#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>

#include "faraday.h"
#include "pc.h"
#include <igl/slice.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/barycentric_coordinates.h>
#include <igl/per_face_normals.h>
#include <igl/parallel_for.h>
#include <igl/marching_tets.h>
#include <igl/AABB.h>
#include <igl/fast_winding_number.h>
#include <igl/knn.h>

#include "geometrycentral/numerical/linear_solvers.h"

Eigen::MatrixXd grad_tets(struct Faraday &f, Eigen::VectorXd &func);

Eigen::MatrixXd grad_tv(struct Faraday &f, Eigen::VectorXd &func);

void build_f_to_v_matrix(struct Faraday &f);

void solvePotentialOverDirs(struct Faraday &f);
void solvePotentialPointCharges(struct Faraday &f, std::vector<int> &pt_constraints, double charge_scale);

void solveMaxFunction(struct Faraday &f);

void estimateNormals(struct Faraday &f);

void classifyInterior(struct Faraday &f, double iso);

void computePointsetFunctions(struct Faraday &f);

Eigen::VectorXd integrateMaxOverStereoPts(struct Faraday &f);

Eigen::VectorXd solveFaraday(struct Faraday &f, geometrycentral::PositiveDefiniteSolver<double> &solver, Eigen::VectorXd &bdry_vals);
Eigen::VectorXd solveFaraday(struct Faraday &f, geometrycentral::PositiveDefiniteSolver<double> &solver, std::unordered_map<int, int> &global_to_matrix_ordering, std::vector<int> &pt_constraints, double const_val);

Eigen::SparseMatrix<double> computeFaraday(struct Faraday &f);
std::tuple<std::unordered_map<int, int>, Eigen::SparseMatrix<double>> computeFaraday(struct Faraday &f, std::vector<int> &pt_constraints);

void faster_igl_quantities(struct Faraday &f);

void extract_isosurface(struct Faraday &f, double isoval);