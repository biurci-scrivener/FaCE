#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"
#include "polyscope/group.h"
#include "faraday.h"

void initPolyscopeStructures(struct Faraday &f);

void reInitCageStructures(struct Faraday &f);

void vis_grad_ranked(struct Faraday &f, int idx);

void vis_u(struct Faraday &f, int idx);

void vis_max(struct Faraday &f);

void vis_interior(struct Faraday &f);

void vis_isosurface(struct Faraday &f);

void vis_normal_results(struct Faraday &f);

void vis_user_charge_locations(struct Faraday &f);

void vis_user_cage_locations(struct Faraday &f);