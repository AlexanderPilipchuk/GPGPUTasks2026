#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    // The extra column avoids local-memory bank conflicts while reading the
    // tile in the transposed direction.
    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    const unsigned int input_x = get_group_id(0) * GROUP_SIZE_X + local_x;
    const unsigned int input_y = get_group_id(1) * GROUP_SIZE_Y + local_y;

    if (input_x < w && input_y < h)
        tile[local_y][local_x] = matrix[input_y * w + input_x];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Swap coordinates inside the tile. Consecutive work-items now write
    // consecutive output elements, so both global reads and writes coalesce.
    const unsigned int output_row = get_group_id(0) * GROUP_SIZE_X + local_y;
    const unsigned int output_col = get_group_id(1) * GROUP_SIZE_Y + local_x;

    if (output_row < w && output_col < h)
        transposed_matrix[output_row * h + output_col] = tile[local_x][local_y];
}
