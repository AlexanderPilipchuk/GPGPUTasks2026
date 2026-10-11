#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float tile_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float tile_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    const unsigned int column = get_global_id(0);
    const unsigned int row = get_global_id(1);

    float sum = 0.0f;
    for (unsigned int tile_start = 0; tile_start < k; tile_start += GROUP_SIZE_X) {
        const unsigned int a_column = tile_start + local_x;
        const unsigned int b_row = tile_start + local_y;

        tile_a[local_y][local_x] =
            (row < h && a_column < k) ? a[row * k + a_column] : 0.0f;
        tile_b[local_y][local_x] =
            (b_row < k && column < w) ? b[b_row * w + column] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int inner = 0; inner < GROUP_SIZE_X; ++inner)
            sum += tile_a[local_y][inner] * tile_b[inner][local_x];

        // No work-item may overwrite a tile while another one still uses it.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (column < w && row < h)
        c[row * w + column] = sum;
}
