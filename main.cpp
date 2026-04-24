#include <iostream>
#include <cuda_runtime.h>

#include "simple_matmul.h"
#include "sample_p_cube.h"
#include "transform.h"
#include "os.h"

#define BLOCK_SIZE 256
#define M 4

int main() {

    // Points sampled on the surface of a cube

    const size_t num_points = 1024; // Number of points to sample on the cube surface
    float points[num_points * 3]; // Each point has 3 coordinates (x, y, z)
    float homo_points[num_points * 4]; // Each homogeneous point has 4 coordinates (x, y, z, w)
    float homo_points_t[4 * num_points]; // Transposed homogeneous points

    sample_cube_surface(points, num_points);
    do_homogeneous_points(points, homo_points, num_points);
    transpose_matrix(homo_points, homo_points_t, num_points, 4);

    to_ply(points, num_points, "debug/init.ply");

    // Device pointers
    float *d_homo_points;

    // Allocate memory for points on the device
    int size_points = num_points * 4 * sizeof(float);
    cudaMalloc(&d_homo_points, size_points);

    // Copy points from host to device
    cudaMemcpy(d_homo_points, homo_points_t, size_points, cudaMemcpyHostToDevice);

    // Generate transformation matrices
    const float angle_x = 30.0f; // Rotation angle around the X-axis in degrees
    const float angle_y = 45.0f; // Rotation angle around the Y-axis in degrees
    const float angle_z = 60.0f; // Rotation angle around the Z-axis in degrees
    const float tx = 1.0f; // Translation along the X-axis
    const float ty = 2.0f; // Translation along the Y-axis
    const float tz = 3.0f; // Translation along the Z-axis
    const float sx = 2.0f; // Scaling factor along the X-axis
    const float sy = 2.0f; // Scaling factor along the Y-axis
    const float sz = 2.0f; // Scaling factor along the Z-axis

    float *h_mat_rot_x, *h_mat_rot_y, *h_mat_rot_z, *h_mat_trans, *h_mat_scale; // Host matrices
    float *d_mat_rot_x, *d_mat_rot_y, *d_mat_rot_z, *d_mat_trans, *d_mat_scale; // Device matrices

    int size = 16 * sizeof(float);

    // Allocate memory for the transformation matrices on the host
    h_mat_rot_x = (float*)malloc(size);
    h_mat_rot_y = (float*)malloc(size);
    h_mat_rot_z = (float*)malloc(size);
    h_mat_trans = (float*)malloc(size);
    h_mat_scale = (float*)malloc(size);

    // Generate the transformation matrices on the host
    generate_rotation_x_matrix(angle_x, h_mat_rot_x);
    generate_rotation_y_matrix(angle_y, h_mat_rot_y);
    generate_rotation_z_matrix(angle_z, h_mat_rot_z);
    generate_translation_matrix(tx, ty, tz, h_mat_trans);
    generate_scaling_matrix(sx, sy, sz, h_mat_scale);

    // Allocate memory for the transformation matrices on the device
    cudaMalloc(&d_mat_rot_x, size);
    cudaMalloc(&d_mat_rot_y, size);
    cudaMalloc(&d_mat_rot_z, size);
    cudaMalloc(&d_mat_trans, size);
    cudaMalloc(&d_mat_scale, size);

    // Copy the transformation matrices from the host to the device
    cudaMemcpy(d_mat_rot_x, h_mat_rot_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_rot_y, h_mat_rot_y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_rot_z, h_mat_rot_z, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_trans, h_mat_trans, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_scale, h_mat_scale, size, cudaMemcpyHostToDevice);

    // Memory allocation for results of transformations on the host and device
    float *h_res_rot_x, *h_res_rot_y, *h_res_rot_z, *h_res_trans, *h_res_scale; // Host results
    float *d_res_rot_x, *d_res_rot_y, *d_res_rot_z, *d_res_trans, *d_res_scale; // Device results

    int size_res = num_points * 4 * sizeof(float);
    
    h_res_rot_x = (float*)malloc(size_res);
    h_res_rot_y = (float*)malloc(size_res);
    h_res_rot_z = (float*)malloc(size_res);
    h_res_trans = (float*)malloc(size_res);
    h_res_scale = (float*)malloc(size_res);

    cudaMalloc(&d_res_rot_x, size_res);
    cudaMalloc(&d_res_rot_y, size_res);
    cudaMalloc(&d_res_rot_z, size_res);
    cudaMalloc(&d_res_trans, size_res);
    cudaMalloc(&d_res_scale, size_res);

    // Rotation around X-axis
    simple_matmul(d_mat_rot_x, d_homo_points, d_res_rot_x, 4, num_points, 4);
    cudaDeviceSynchronize();

    // Copy results from device to host
    cudaMemcpy(h_res_rot_x, d_res_rot_x, size_res, cudaMemcpyDeviceToHost);
    
    // Transpose result
    float* h_res_rot_x_t = (float*)malloc(size_res);
    transpose_matrix(h_res_rot_x, h_res_rot_x_t, 4, num_points);

    // Undo homogeneous coordinates
    undo_homogeneous_points(h_res_rot_x, h_res_rot_x_t, num_points);

    // Save transformed points to PLY file
    to_ply(h_res_rot_x, num_points, "debug/rot_x.ply");

    // Rotation around Y-axis
    simple_matmul(d_mat_rot_y, d_res_rot_x, d_res_rot_y, 4, num_points, 4);
    cudaDeviceSynchronize();

    // Copy results from device to host
    cudaMemcpy(h_res_rot_y, d_res_rot_y, size_res, cudaMemcpyDeviceToHost);

    // Transpose result
    float* h_res_rot_y_t = (float*)malloc(size_res);
    transpose_matrix(h_res_rot_y, h_res_rot_y_t, 4, num_points);

    // Undo homogeneous coordinates
    undo_homogeneous_points(h_res_rot_y, h_res_rot_y_t, num_points);

    // Save transformed points to PLY file
    to_ply(h_res_rot_y, num_points, "debug/rot_x_y.ply");

    // Rotation around Z-axis
    simple_matmul(d_mat_rot_z, d_res_rot_y, d_res_rot_z, 4, num_points, 4);
    cudaDeviceSynchronize();

    // Copy results from device to host
    cudaMemcpy(h_res_rot_z, d_res_rot_z, size_res, cudaMemcpyDeviceToHost);

    // Transpose result
    float* h_res_rot_z_t = (float*)malloc(size_res);
    transpose_matrix(h_res_rot_z, h_res_rot_z_t, 4, num_points);

    // Undo homogeneous coordinates
    undo_homogeneous_points(h_res_rot_z, h_res_rot_z_t, num_points);

    // Save transformed points to PLY file
    to_ply(h_res_rot_z, num_points, "debug/rot_x_y_z.ply");

    // Translation
    simple_matmul(d_mat_trans, d_res_rot_z, d_res_trans, 4, num_points, 4);
    cudaDeviceSynchronize();

    // Copy results from device to host
    cudaMemcpy(h_res_trans, d_res_trans, size_res, cudaMemcpyDeviceToHost);

    // Transpose result
    float* h_res_trans_t = (float*)malloc(size_res);
    transpose_matrix(h_res_trans, h_res_trans_t, 4, num_points);

    // Undo homogeneous coordinates
    undo_homogeneous_points(h_res_trans, h_res_trans_t, num_points);

    // Save transformed points to PLY file
    to_ply(h_res_trans, num_points, "debug/trans.ply");

    // Scaling
    simple_matmul(d_mat_scale, d_res_trans, d_res_scale, 4, num_points, 4);
    cudaDeviceSynchronize();

    // Copy results from device to host
    cudaMemcpy(h_res_scale, d_res_scale, size_res, cudaMemcpyDeviceToHost);

    // Transpose result
    float* h_res_scale_t = (float*)malloc(size_res);
    transpose_matrix(h_res_scale, h_res_scale_t, 4, num_points);

    // Undo homogeneous coordinates
    undo_homogeneous_points(h_res_scale, h_res_scale_t, num_points);

    // Save transformed points to PLY file
    to_ply(h_res_scale, num_points, "debug/scale.ply");

    // Fused Transformation (Rotation + Translation + Scaling)

    // Fused Transformation Matrix: M = S * T * Rz * Ry * Rx
    float *h_mat_fused, *h_res_fused;
    float *d_mat_fused, *d_res_fused;

    h_mat_fused = (float*)malloc(size);
    h_res_fused = (float*)malloc(size_res);

    generate_fused_transform_matrix(tx, ty, tz, angle_x, angle_y, angle_z, sx, sy, sz, h_mat_fused);
    
    cudaMalloc(&d_mat_fused, size);
    cudaMemcpy(d_mat_fused, h_mat_fused, size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_res_fused, size_res);
    simple_matmul(d_mat_fused, d_homo_points, d_res_fused, 4, num_points, 4);
    cudaDeviceSynchronize();

    cudaMemcpy(h_res_fused, d_res_fused, size_res, cudaMemcpyDeviceToHost);

    float* h_res_fused_t = (float*)malloc(size_res);
    transpose_matrix(h_res_fused, h_res_fused_t, 4, num_points);

    undo_homogeneous_points(h_res_fused, h_res_fused_t, num_points);

    to_ply(h_res_fused, num_points, "debug/fused.ply");

    // Correctness

    bool correct = true;

    for (size_t i = 0; i < num_points; ++i) {
        if (fabs(h_res_fused[i] - h_res_scale[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Fused transformation is correct." << std::endl;
    } else {
        std::cout << "Fused transformation is incorrect." << std::endl;
    }

    // Clear memory

    free(h_res_rot_x);
    free(h_res_rot_y);
    free(h_res_rot_z);
    free(h_res_trans);
    free(h_res_scale);
    free(h_res_trans_t);
    free(h_mat_fused);
    free(h_res_fused);
    free(h_res_fused_t);

    cudaFree(d_res_rot_x);
    cudaFree(d_res_rot_y);
    cudaFree(d_res_rot_z);
    cudaFree(d_res_trans);
    cudaFree(d_res_scale);
    cudaFree(d_mat_rot_x);
    cudaFree(d_mat_rot_y);
    cudaFree(d_mat_rot_z);
    cudaFree(d_mat_trans);
    cudaFree(d_mat_scale);
    cudaFree(d_mat_fused);
    cudaFree(d_res_fused);
}