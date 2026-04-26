#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#include "csrc/simple_matmul.h"
#include "csrc/sample_p_cube.h"
#include "csrc/transform.h"
#include "csrc/utils.h"
#include "csrc/os.h"

#define BLOCK_SIZE 256
#define M 4

using namespace std;

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {

    // Points sampled on the surface of a cube

    const size_t num_points = 100000000; // Number of points to sample on the cube surface
    float *points = (float*)malloc(num_points * 3 * sizeof(float)); // Each point has 3 coordinates (x, y, z)
    float *homo_points = (float*)malloc(num_points * 4 * sizeof(float)); // Each homogeneous point has 4 coordinates (x, y, z, w)
    float *homo_points_t = (float*)malloc(4 * num_points * sizeof(float)); // Transposed homogeneous points
    cudaStream_t compute_stream, data_stream_1, data_stream_2, data_stream_3, data_stream_4, data_stream_5;

    if (points == NULL || homo_points == NULL || homo_points_t == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    CHECK_CUDA_ERROR(cudaStreamCreate(&compute_stream));
    CHECK_CUDA_ERROR(cudaStreamCreate(&data_stream_1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&data_stream_2));
    CHECK_CUDA_ERROR(cudaStreamCreate(&data_stream_3));
    CHECK_CUDA_ERROR(cudaStreamCreate(&data_stream_4));
    CHECK_CUDA_ERROR(cudaStreamCreate(&data_stream_5));

    sample_cube_surface(points, num_points);
    do_homogeneous_points(points, homo_points, num_points);
    transpose_matrix(homo_points, homo_points_t, num_points, 4);

    // Device pointers
    float *d_homo_points;

    // Allocate memory for points on the device
    size_t size_points = num_points * 4 * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_homo_points, size_points));

    // Copy points from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_homo_points, homo_points_t, size_points, cudaMemcpyHostToDevice));

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

    size_t size = 16 * sizeof(float);

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
    CHECK_CUDA_ERROR(cudaMalloc(&d_mat_rot_x, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_mat_rot_y, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_mat_rot_z, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_mat_trans, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_mat_scale, size));

    // Copy the transformation matrices from the host to the device asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_rot_x, h_mat_rot_x, size, cudaMemcpyHostToDevice, data_stream_1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_rot_y, h_mat_rot_y, size, cudaMemcpyHostToDevice, data_stream_2));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_rot_z, h_mat_rot_z, size, cudaMemcpyHostToDevice, data_stream_3));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_trans, h_mat_trans, size, cudaMemcpyHostToDevice, data_stream_4));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_mat_scale, h_mat_scale, size, cudaMemcpyHostToDevice, data_stream_5));

    // Memory allocation for results of transformations on the host and device
    float *h_res_rot_x, *h_res_rot_y, *h_res_rot_z, *h_res_trans, *h_res_scale; // Host results
    float *d_res_rot_x, *d_res_rot_y, *d_res_rot_z, *d_res_trans, *d_res_scale; // Device results

    size_t size_res = num_points * 4 * sizeof(float);
    
    h_res_rot_x = (float*)malloc(size_res);
    h_res_rot_y = (float*)malloc(size_res);
    h_res_rot_z = (float*)malloc(size_res);
    h_res_trans = (float*)malloc(size_res);
    h_res_scale = (float*)malloc(size_res);

    CHECK_CUDA_ERROR(cudaMalloc(&d_res_rot_x, size_res));
    CHECK_CUDA_ERROR(cudaMalloc(&d_res_rot_y, size_res));
    CHECK_CUDA_ERROR(cudaMalloc(&d_res_rot_z, size_res));
    CHECK_CUDA_ERROR(cudaMalloc(&d_res_trans, size_res));
    CHECK_CUDA_ERROR(cudaMalloc(&d_res_scale, size_res));

    // Non-fused Transformations
    double non_fused_total_time = 0.0;
    double start_time = get_time();

    // Rotation around X-axis
    simple_matmul(d_mat_rot_x, d_homo_points, d_res_rot_x, 4, num_points, 4, compute_stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(compute_stream));
    
    // Rotation around Y-axis
    simple_matmul(d_mat_rot_y, d_res_rot_x, d_res_rot_y, 4, num_points, 4, compute_stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(compute_stream));

    // Rotation around Z-axis
    simple_matmul(d_mat_rot_z, d_res_rot_y, d_res_rot_z, 4, num_points, 4, compute_stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(compute_stream));

    simple_matmul(d_mat_trans, d_res_rot_z, d_res_trans, 4, num_points, 4, compute_stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(compute_stream));

    simple_matmul(d_mat_scale, d_res_trans, d_res_scale, 4, num_points, 4, compute_stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(compute_stream));

    double end_time = get_time();
    non_fused_total_time = end_time - start_time;

    // Fused Transformations
    float *h_mat_fused, *h_res_fused;
    float *d_mat_fused, *d_res_fused;

    h_mat_fused = (float*)malloc(size);
    h_res_fused = (float*)malloc(size_res);

    generate_fused_transform_matrix(tx, ty, tz, angle_x, angle_y, angle_z, sx, sy, sz, h_mat_fused);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_mat_fused, size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_mat_fused, h_mat_fused, size, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMalloc(&d_res_fused, size_res));

    double fused_total_time = 0.0;
    double start_time_fused = get_time();

    simple_matmul(d_mat_fused, d_homo_points, d_res_fused, 4, num_points, 4, compute_stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(compute_stream));

    double end_time_fused = get_time();
    fused_total_time = end_time_fused - start_time_fused;

    cout << "Non-fused total time: " << non_fused_total_time << " ms" << endl;
    cout << "Fused total time: " << fused_total_time << " ms" << endl;
    cout << "Speedup: " << non_fused_total_time / fused_total_time << "x" << endl;

    // Clear memory

    free(h_res_rot_x);
    free(h_res_rot_y);
    free(h_res_rot_z);
    free(h_res_trans);
    free(h_res_scale);
    free(h_mat_fused);
    free(h_res_fused);

    CHECK_CUDA_ERROR(cudaFree(d_res_rot_x));
    CHECK_CUDA_ERROR(cudaFree(d_res_rot_y));
    CHECK_CUDA_ERROR(cudaFree(d_res_rot_z));
    CHECK_CUDA_ERROR(cudaFree(d_res_trans));
    CHECK_CUDA_ERROR(cudaFree(d_res_scale));
    CHECK_CUDA_ERROR(cudaFree(d_mat_rot_x));
    CHECK_CUDA_ERROR(cudaFree(d_mat_rot_y));
    CHECK_CUDA_ERROR(cudaFree(d_mat_rot_z));
    CHECK_CUDA_ERROR(cudaFree(d_mat_trans));
    CHECK_CUDA_ERROR(cudaFree(d_mat_scale));
    CHECK_CUDA_ERROR(cudaFree(d_mat_fused));
    CHECK_CUDA_ERROR(cudaFree(d_res_fused));
    CHECK_CUDA_ERROR(cudaStreamDestroy(compute_stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(data_stream_1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(data_stream_2));
    CHECK_CUDA_ERROR(cudaStreamDestroy(data_stream_3));
    CHECK_CUDA_ERROR(cudaStreamDestroy(data_stream_4));
    CHECK_CUDA_ERROR(cudaStreamDestroy(data_stream_5));

}