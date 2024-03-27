// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "Registration.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>

#include "VoxelHashMap.hpp"

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen
using Associations = std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>;
using LinearSystem = std::pair<Eigen::Matrix6d, Eigen::Vector6d>;

namespace {
inline double square(double x) { return x * x; }

void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { return T * point; });
}

using Voxel = kiss_icp::VoxelHashMap::Voxel;
std::vector<Voxel> GetAdjacentVoxels(const Voxel &voxel, int adjacent_voxels = 1) {
    std::vector<Voxel> voxel_neighborhood;
    for (int i = voxel.x() - adjacent_voxels; i < voxel.x() + adjacent_voxels + 1; ++i) {
        for (int j = voxel.y() - adjacent_voxels; j < voxel.y() + adjacent_voxels + 1; ++j) {
            for (int k = voxel.z() - adjacent_voxels; k < voxel.z() + adjacent_voxels + 1; ++k) {
                voxel_neighborhood.emplace_back(i, j, k);
            }
        }
    }
    return voxel_neighborhood;
}

std::tuple<Eigen::Vector3d, double> GetClosestNeighbor(const Eigen::Vector3d &point,
                                                       const kiss_icp::VoxelHashMap &voxel_map) {
    // Convert the point to voxel coordinates
    const auto &voxel = voxel_map.PointToVoxel(point);
    // Get nearby voxels on the map
    const auto &query_voxels = GetAdjacentVoxels(voxel);
    // Extract the points contained within the neighborhood voxels
    const auto &neighbors = voxel_map.GetPoints(query_voxels);

    // Find the nearest neighbor
    Eigen::Vector3d closest_neighbor;
    double closest_distance = std::numeric_limits<double>::max();
    std::for_each(neighbors.cbegin(), neighbors.cend(), [&](const auto &neighbor) {
        double distance = (neighbor - point).norm();
        if (distance < closest_distance) {
            closest_neighbor = neighbor;
            closest_distance = distance;
        }
    });
    return std::make_tuple(closest_neighbor, closest_distance);
}

Associations FindAssociations(const std::vector<Eigen::Vector3d> &points,
                              const kiss_icp::VoxelHashMap &voxel_map,
                              double max_correspondance_distance) {
    Associations associations;
    associations.reserve(points.size());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for(size_t i = 0; i < points.size(); i++) {
        const auto &[closest_neighbor, distance] = GetClosestNeighbor(points[i], voxel_map);
        if (distance < max_correspondance_distance) {
            #pragma omp critical
            associations.emplace_back(points[i], closest_neighbor);
        }
    }

    return associations;
}

LinearSystem BuildLinearSystem(const Associations &associations, double kernel) {
    auto compute_jacobian_and_residual = [](auto association) {
        const auto &[source, target] = association;
        const Eigen::Vector3d residual = source - target;
        Eigen::Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source);
        return std::make_tuple(J_r, residual);
    };

    auto GM_weight = [&](double residual2) { return square(kernel) / square(kernel + residual2); };

    LinearSystem J(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for(size_t i = 0; i < associations.size(); i++) {
        const auto &[J_r, residual] = compute_jacobian_and_residual(associations[i]);
        const double w = GM_weight(residual.squaredNorm());
        LinearSystem temp(J_r.transpose() * w * J_r,        // JTJ
                          J_r.transpose() * w * residual);  // JTr
        #pragma omp critical
        {
            J.first += temp.first;
            J.second += temp.second;
        }
    }

    return J;
}
}  // namespace

namespace kiss_icp {

Registration::Registration(int max_num_iteration, double convergence_criterion, int max_num_threads)
    : max_num_iterations_(max_num_iteration),
      convergence_criterion_(convergence_criterion),
      max_num_threads_(max_num_threads > 0 ? max_num_threads : omp_get_max_threads()) {
    omp_set_num_threads(max_num_threads_);
}

Sophus::SE3d Registration::AlignPointsToMap(const std::vector<Eigen::Vector3d> &frame,
                                            const VoxelHashMap &voxel_map,
                                            const Sophus::SE3d &initial_guess,
                                            double max_correspondence_distance,
                                            double kernel) {
    if (voxel_map.Empty()) return initial_guess;

    // Equation (9)
    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    // ICP-loop
    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < max_num_iterations_; ++j) {
        // Equation (10)
        const auto associations = FindAssociations(source, voxel_map, max_correspondence_distance);
        // Equation (11)
        const auto &[JTJ, JTr] = BuildLinearSystem(associations, kernel);
        const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        // Equation (12)
        TransformPoints(estimation, source);
        // Update iterations
        T_icp = estimation * T_icp;
        // Termination criteria
        if (dx.norm() < convergence_criterion_) break;
    }
    // Spit the final transformation
    return T_icp * initial_guess;
}

}  // namespace kiss_icp
