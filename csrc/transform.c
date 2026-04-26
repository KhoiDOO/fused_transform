#include <transform.h>
#include <stdlib.h>

void generate_rotation_x_matrix(float angle, float* matrix) {
    float radians = angle * 3.14159265f / 180.0f;
    float cos_a = cosf(radians);
    float sin_a = sinf(radians);

    matrix[0] = 1; matrix[1] = 0;     matrix[2] = 0;      matrix[3] = 0;
    matrix[4] = 0; matrix[5] = cos_a; matrix[6] = -sin_a; matrix[7] = 0;
    matrix[8] = 0; matrix[9] = sin_a; matrix[10] = cos_a; matrix[11] = 0;
    matrix[12] = 0; matrix[13] = 0;   matrix[14] = 0;     matrix[15] = 1;
}

void generate_rotation_y_matrix(float angle, float* matrix) {
    float radians = angle * 3.14159265f / 180.0f;
    float cos_a = cosf(radians);
    float sin_a = sinf(radians);

    matrix[0] = cos_a;  matrix[1] = 0; matrix[2] = sin_a; matrix[3] = 0;
    matrix[4] = 0;      matrix[5] = 1; matrix[6] = 0;     matrix[7] = 0;
    matrix[8] = -sin_a; matrix[9] = 0; matrix[10] = cos_a; matrix[11] = 0;
    matrix[12] = 0;     matrix[13] = 0; matrix[14] = 0;   matrix[15] = 1;
}

void generate_rotation_z_matrix(float angle, float* matrix) {
    float radians = angle * 3.14159265f / 180.0f;
    float cos_a = cosf(radians);
    float sin_a = sinf(radians);

    matrix[0] = cos_a; matrix[1] = -sin_a; matrix[2] = 0; matrix[3] = 0;
    matrix[4] = sin_a; matrix[5] = cos_a;  matrix[6] = 0; matrix[7] = 0;
    matrix[8] = 0;     matrix[9] = 0;     matrix[10] = 1; matrix[11] = 0;
    matrix[12] = 0;    matrix[13] = 0;    matrix[14] = 0;   matrix[15] = 1;
}

void generate_translation_matrix(float tx, float ty, float tz, float* matrix) {
    matrix[0] = 1; matrix[1] = 0; matrix[2] = 0; matrix[3] = tx;
    matrix[4] = 0; matrix[5] = 1; matrix[6] = 0; matrix[7] = ty;
    matrix[8] = 0; matrix[9] = 0; matrix[10] = 1; matrix[11] = tz;
    matrix[12] = 0; matrix[13] = 0; matrix[14] = 0; matrix[15] = 1;
}

void generate_scaling_matrix(float sx, float sy, float sz, float* matrix) {
    matrix[0] = sx; matrix[1] = 0;  matrix[2] = 0;  matrix[3] = 0;
    matrix[4] = 0;  matrix[5] = sy; matrix[6] = 0;  matrix[7] = 0;
    matrix[8] = 0;  matrix[9] = 0;  matrix[10] = sz; matrix[11] = 0;
    matrix[12] = 0; matrix[13] = 0; matrix[14] = 0;  matrix[15] = 1;
}

void generate_fused_transform_matrix(float tx, float ty, float tz, float angle_x, float angle_y, float angle_z, float sx, float sy, float sz, float* matrix) {
    
    float rad_x = angle_x * 3.14159265f / 180.0f;
    float rad_y = angle_y * 3.14159265f / 180.0f;
    float rad_z = angle_z * 3.14159265f / 180.0f;

    float cos_x = cosf(rad_x);
    float sin_x = sinf(rad_x);
    float cos_y = cosf(rad_y);
    float sin_y = sinf(rad_y);
    float cos_z = cosf(rad_z);
    float sin_z = sinf(rad_z);

    matrix[0] = sx * (cos_y * cos_z); 
    matrix[1] = sx * (cos_z * sin_y * sin_x - sin_z * cos_x); 
    matrix[2] = sx * (cos_z * sin_y * cos_x + sin_z * sin_x); 
    matrix[3] = sx * tx;
    
    matrix[4] = sy * (sin_z * cos_y);
    matrix[5] = sy * (sin_z * sin_y * sin_x + cos_z * cos_x);
    matrix[6] = sy * (sin_z * sin_y * cos_x - cos_z * sin_x);
    matrix[7] = sy * ty;
    
    matrix[8] = sz * (-sin_y);
    matrix[9] = sz * (cos_y * sin_x);
    matrix[10] = sz * (cos_y * cos_x);
    matrix[11] = sz * tz;
    
    matrix[12] = 0;
    matrix[13] = 0;
    matrix[14] = 0;
    matrix[15] = 1;
}

void do_homogeneous_points(float* points, float* homogeneous_points, size_t num_points) {
    for (size_t i = 0; i < num_points; ++i) {
        homogeneous_points[i * 4 + 0] = points[i * 3 + 0];
        homogeneous_points[i * 4 + 1] = points[i * 3 + 1];
        homogeneous_points[i * 4 + 2] = points[i * 3 + 2];
        homogeneous_points[i * 4 + 3] = 1.0f;
    }
}

void undo_homogeneous_points(float* points, float* homogeneous_points, size_t num_points) {
    for (size_t i = 0; i < num_points; ++i) {
        float w = homogeneous_points[i * 4 + 3];
        if (w == 0) {
            points[i * 3 + 0] = homogeneous_points[i * 4 + 0];
            points[i * 3 + 1] = homogeneous_points[i * 4 + 1];
            points[i * 3 + 2] = homogeneous_points[i * 4 + 2];
        } else {
            points[i * 3 + 0] = homogeneous_points[i * 4 + 0] / w;
            points[i * 3 + 1] = homogeneous_points[i * 4 + 1] / w;
            points[i * 3 + 2] = homogeneous_points[i * 4 + 2] / w;
        }
    }
}

void transpose_matrix(const float* matrix, float* transposed_matrix, size_t rows, size_t cols) {
    for (size_t i =
         0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed_matrix[j * rows + i] = matrix[i * cols + j];
        }
    }
}