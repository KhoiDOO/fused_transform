#include "os.h"
#include <stdio.h>
#include <stdlib.h>


void to_ply(const float *points, size_t num_points, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Failed to open file for writing");
        return;
    }

    if (points == NULL || num_points == 0) {
        fclose(file);
        return;
    }

    fprintf(file, "ply\nformat ascii 1.0\nelement vertex %zu\nproperty float x\nproperty float y\nproperty float z\nend_header\n", num_points);
    for (size_t i = 0; i < num_points; ++i) {
        fprintf(file, "%f %f %f\n", points[3 * i], points[3 * i + 1], points[3 * i + 2]);
    }

    fclose(file);
}