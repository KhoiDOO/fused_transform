#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void generate_rotation_x_matrix(float angle, float* matrix);

void generate_rotation_y_matrix(float angle, float* matrix);

void generate_rotation_z_matrix(float angle, float* matrix);

void generate_translation_matrix(float tx, float ty, float tz, float* matrix);

void generate_scaling_matrix(float sx, float sy, float sz, float* matrix);

void generate_fused_transform_matrix(float tx, float ty, float tz, float angle_x, float angle_y, float angle_z, float sx, float sy, float sz, float* matrix);

void do_homogeneous_points(float* points, float* homogeneous_points, size_t num_points);

void undo_homogeneous_points(float* points, float* homogeneous_points, size_t num_points);

void transpose_matrix(const float* matrix, float* transposed_matrix, size_t rows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif // TRANSFORM_H