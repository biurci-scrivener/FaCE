#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <set>

#include <Eigen/Dense>

#include "faraday.h"


std::tuple<Eigen::MatrixXd, std::vector<int>> removeDuplicatePoints(const Eigen::MatrixXd &points) {
    std::vector<std::pair<Eigen::Vector3d, int>> pointVec;
    for (int i = 0; i < points.rows(); ++i) {
        pointVec.push_back({points.row(i), i});
    }

    std::set<std::pair<Eigen::Vector3d, int>, 
             bool(*)(const std::pair<Eigen::Vector3d, int>&, const std::pair<Eigen::Vector3d, int>&)> uniquePoints(
                 [](const std::pair<Eigen::Vector3d, int>& a, const std::pair<Eigen::Vector3d, int>& b) {
                     return std::tie(a.first[0], a.first[1], a.first[2]) < std::tie(b.first[0], b.first[1], b.first[2]);
                 });

    std::vector<int> removedIndices;
    removedIndices.clear();
    for (int i = 0; i < pointVec.size(); ++i) {
        auto result = uniquePoints.insert(pointVec[i]);
        if (!result.second) {  // if insertion failed, it's a duplicate
            removedIndices.push_back(pointVec[i].second); 
        }
    }

    std::vector<std::pair<Eigen::Vector3d, int>> uniquePointVec(uniquePoints.begin(), uniquePoints.end());
    std::sort(uniquePointVec.begin(), uniquePointVec.end(), [](const std::pair<Eigen::Vector3d, int>& a, const std::pair<Eigen::Vector3d, int>& b) {return (a.second < b.second);});

    Eigen::MatrixXd uniquePointsMat(uniquePointVec.size(), 3);
    for (size_t i = 0; i < uniquePointVec.size(); ++i) {
        uniquePointsMat.row(i) = uniquePointVec[i].first;
    }

    std::cout << "Handled " << removedIndices.size() << " duplicates" << std::endl;
    return std::make_tuple(uniquePointsMat, removedIndices);
}


Eigen::MatrixXd removeDuplicateNormals(const Eigen::MatrixXd &normals, std::vector<int> dup_ids) {

    std::vector<int> nonRemovedIndices;
    for (int i = 0; i < normals.rows(); ++i) {
        if (std::find(dup_ids.begin(), dup_ids.end(), i) == dup_ids.end()) {
            nonRemovedIndices.push_back(i);
        }
    }

    Eigen::MatrixXd cleanedNormals(nonRemovedIndices.size(), normals.cols());
    for (size_t i = 0; i < nonRemovedIndices.size(); ++i) {
        cleanedNormals.row(i) = normals.row(nonRemovedIndices[i]);
    }

    return cleanedNormals;

}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> parseXYZ(std::string &filename) {
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Error reading file");
    }

    std::string line;
    bool inHeader = true;
    int vertexCount = 0;

    std::vector<Eigen::VectorXd> points;
    std::vector<Eigen::VectorXd> normals;

    // Read vertex data
    while(std::getline(file, line)) {
        
        std::stringstream ss(line);

        double x, y, z;
        if (ss >> x >> y >> z) {
            points.push_back(Eigen::Vector3d(x, y, z));
        } else {
            continue;
        }

        // Check for normals
        double nx, ny, nz;
        if (ss >> nx >> ny >> nz) {
            // some of the GCNO models have non-unit normals
            normals.push_back(Eigen::Vector3d(nx, ny, nz).normalized());
        } else {
            normals.push_back(Eigen::Vector3d(0.,0.,0.));
        }

    }

    file.close();

    Eigen::MatrixXd V(points.size(), 3);
    Eigen::MatrixXd N(points.size(), 3);

    for (int i = 0; i < points.size(); i++) {
        V.row(i) = points[i];
        N.row(i) = normals[i];
    }

    Eigen::MatrixXd P_cleaned;
    std::vector<int> pt_dups;

    std::tie(P_cleaned, pt_dups) = removeDuplicatePoints(V);


    return std::make_tuple(P_cleaned, removeDuplicateNormals(N, pt_dups));
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> parseOBJ(const std::string& filename) {
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Error reading file");
    }

    std::string line;
    std::vector<Eigen::VectorXd> points;
    std::vector<Eigen::VectorXd> normals;

    // Read the file line by line
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;

        // Parse vertices
        if (line.rfind("v ", 0) == 0) {
            double x, y, z;
            iss >> type >> x >> y >> z;
            points.push_back(Eigen::Vector3d(x, y, z));
        }
        // Parse vertex normals
        else if (line.rfind("vn ", 0) == 0) {
            double nx, ny, nz;
            iss >> type >> nx >> ny >> nz;
            normals.push_back(Eigen::Vector3d(nx, ny, nz).normalized());
        }
    }

    file.close();

    // Create Eigen matrices from the vectors
    Eigen::MatrixXd V(points.size(), 3);
    Eigen::MatrixXd N(normals.size(), 3);

    // Fill the matrices
    for (size_t i = 0; i < points.size(); ++i) {
        V.row(i) = points[i];
    }

    // If there are no normals, assume zero normals
    if (normals.empty()) {
        N = Eigen::MatrixXd::Zero(points.size(), 3);
    } else {
        for (size_t i = 0; i < normals.size(); ++i) {
            N.row(i) = normals[i];
        }
    }

    Eigen::MatrixXd P_cleaned;
    std::vector<int> pt_dups;

    std::tie(P_cleaned, pt_dups) = removeDuplicatePoints(V);


    return std::make_tuple(P_cleaned, removeDuplicateNormals(N, pt_dups));
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> parsePLY(std::string &filename) {
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Error reading file");
    }

    std::string line;
    bool inHeader = true;
    int vertexCount = 0;

    // Read the header
    while (inHeader && std::getline(file, line)) {
        if (line.rfind("end_header", 0) == 0) {
            inHeader = false;
            continue;
        }

        if (line.rfind("element vertex", 0) == 0) {
            std::istringstream iss(line);
            std::string element;
            iss >> element >> element >> vertexCount;
        } 
    }

    Eigen::MatrixXd V(vertexCount, 3);
    Eigen::MatrixXd N(vertexCount, 3);

    // Read vertex data
    for (int i = 0; i < vertexCount; ++i) {
        
        double x, y, z;
        file >> x >> y >> z;
        V.row(i) << x, y, z;

        // Check for normals
        if (file.peek() != '\n') {
            double xn, yn, zn;
            file >> xn >> yn >> zn;
            N.row(i) << xn, yn, zn;
        } else {
            N.row(i) << 0., 0., 0.;
        }

        // Skip any extra attributes on the same line after the normal
        std::string extraAttributes;
        std::getline(file, extraAttributes);  // Read the rest of the line

    }

    file.close();

    Eigen::MatrixXd P_cleaned;
    std::vector<int> pt_dups;

    std::tie(P_cleaned, pt_dups) = removeDuplicatePoints(V);


    return std::make_tuple(P_cleaned, removeDuplicateNormals(N, pt_dups));
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> my_load_points(std::string &filename) {

    /*
        Seems like igl::read_triangle_mesh won't pull normals from the points
    */

    std::filesystem::path path(filename);
    std::string ext = path.extension().string();

    if (ext.empty()) {
        throw std::runtime_error("Input file has no extension");
    } else if ((ext == ".ply") || (ext == ".PLY")) {
        return parsePLY(filename);
    } else if ((ext == ".obj") || (ext == ".OBJ")) {
        return parseOBJ(filename);
    } else if ((ext == ".xyz") || (ext == ".XYZ")) {
        return parseXYZ(filename);
    } else {
        throw std::runtime_error("Unknown file extension. Acceptable formats are XYZ, PLY, and OBJ");
    }

}

int saveXYZ(struct Faraday & f, std::string &filename, bool keep_interior) {

    std::ofstream file(filename);
    
    if (!file) {
        std::cerr << "Error writing file" << filename << std::endl;
        return -1;
    }

    for (size_t i = 0; i < f.P.rows(); i++) {
        if (f.is_interior(i) && !keep_interior) {
            continue;
        }
        file << f.P(i, 0) << " "
                << f.P(i, 1) << " "
                << f.P(i, 2) << " "
                << f.N_est(i, 0) << " "
                << f.N_est(i, 1) << " "
                << f.N_est(i, 2) << "\n";
    }

    file.close();

    std::cout << "XYZ file written to: " << filename << std::endl;

    return 0;

}

void savePLY(struct Faraday & f, const std::string& filename, bool keep_interior) {


    // Open file for writing
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    int num_vertices = f.P.rows();
    
    // Write the PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    if (keep_interior) {
        file << "element vertex " << num_vertices << "\n";
    } else {
        file << "element vertex " << num_vertices - f.is_interior.sum() << "\n";
    }
    
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float nx\n";
    file << "property float ny\n";
    file << "property float nz\n";
    file << "property float quality\n";
    file << "end_header\n";
    
    // Write vertices, normals, and quality values
    for (int i = 0; i < num_vertices; ++i) {
        if (!keep_interior && (f.is_interior(i))) {
            continue;
        }
        file << f.P(i, 0) << " " << f.P(i, 1) << " " << f.P(i, 2) << " ";  // Write x, y, z
        file << f.N_est(i, 0) << " " << f.N_est(i, 1) << " " << f.N_est(i, 2) << " ";  // Write nx, ny, nz
        file << fmin(f.P_avg_value(i), 1.) << "\n";  // Write quality value
    }

    file.close();
    std::cout << "PLY file written to: " << filename << std::endl;
}