#ifndef MAX_BOUNDS
#define MAX_BOUNDS 0.5f
#endif

#ifndef MIN_BOUNDS
#define MIN_BOUNDS -0.5f
#endif

#ifndef VERTICES
#define VERTICES {\
	{MIN_BOUNDS, MIN_BOUNDS, MIN_BOUNDS},\
	{MAX_BOUNDS, MIN_BOUNDS, MIN_BOUNDS},\
	{MAX_BOUNDS, MAX_BOUNDS, MIN_BOUNDS},\
	{MIN_BOUNDS, MAX_BOUNDS, MIN_BOUNDS},\
	{MIN_BOUNDS, MIN_BOUNDS, MAX_BOUNDS},\
	{MAX_BOUNDS, MIN_BOUNDS, MAX_BOUNDS},\
	{MAX_BOUNDS, MAX_BOUNDS, MAX_BOUNDS},\
	{MIN_BOUNDS, MAX_BOUNDS, MAX_BOUNDS}\
}
#endif

#ifndef INDICES
#define INDICES {\
	{0, 1, 2}, {0, 2, 3}, /* Back face */\
	{4, 5, 6}, {4, 6, 7}, /* Front face */\
	{0, 1, 5}, {0, 5, 4}, /* Bottom face */\
	{2, 3, 7}, {2, 7, 6}, /* Top face */\
	{1, 2, 6}, {1, 6, 5}, /* Right face */\
	{0, 3, 7}, {0, 7, 4}  /* Left face */\
}
#endif

#include "sample_p_cube.h"
#include <math.h>
#include <stdlib.h>

static void sample_point_on_triangle(const float a[3], const float b[3], const float c[3], float out[3]) {
	float r1 = (float)rand() / (float)RAND_MAX;
	float r2 = (float)rand() / (float)RAND_MAX;

	if (r1 + r2 > 1.0f) {
		r1 = 1.0f - r1;
		r2 = 1.0f - r2;
	}
	
	float w0 = 1 - r1 - r2;
	float w1 = r1;
	float w2 = r2;

	out[0] = (w0 * a[0]) + (w1 * b[0]) + (w2 * c[0]);
	out[1] = (w0 * a[1]) + (w1 * b[1]) + (w2 * c[1]);
	out[2] = (w0 * a[2]) + (w1 * b[2]) + (w2 * c[2]);
}

void sample_cube_surface(float *points, size_t count) {
	if (points == NULL || count == 0) {
		return;
	}

	// Define the vertices of a unit cube centered at the origin
	float vertices[8][3] = VERTICES;

	// Define 12 triangles (2 per face) over the cube surface
	int indices[12][3] = INDICES;

	int num_triangles = 12;

	for (size_t i = 0; i < count; ++i) {
		int tri_idx = rand() % num_triangles;
		int i0 = indices[tri_idx][0];
		int i1 = indices[tri_idx][1];
		int i2 = indices[tri_idx][2];

		float p[3];
		sample_point_on_triangle(vertices[i0], vertices[i1], vertices[i2], p);

		points[(i * 3) + 0] = p[0];
		points[(i * 3) + 1] = p[1];
		points[(i * 3) + 2] = p[2];
	}
}