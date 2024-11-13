from numba import cuda
import numpy as np

@cuda.jit
def whoami():
    # Thread and block indices
    block_id = (cuda.blockIdx.x + cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.z * cuda.gridDim.x * cuda.gridDim.y)
    block_offset = block_id * cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
    thread_offset = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y

    # Global thread ID
    id = block_offset + thread_offset

    # Printing the output
    print(f"{id:04d} | Block({cuda.blockIdx.x} {cuda.blockIdx.y} {cuda.blockIdx.z}) = {block_id:3d} | Thread({cuda.threadIdx.x} {cuda.threadIdx.y} {cuda.threadIdx.z}) = {thread_offset:3d}")

def main():
    b_x, b_y, b_z = 2, 3, 4
    t_x, t_y, t_z = 4, 4, 4

    blocks_per_grid = b_x * b_y * b_z
    threads_per_block = t_x * t_y * t_z

    print(f"{blocks_per_grid} blocks/grid")
    print(f"{threads_per_block} threads/block")
    print(f"{blocks_per_grid * threads_per_block} total threads")

    # Launching the kernel
    blocks_per_grid = (b_x, b_y, b_z)
    threads_per_block = (t_x, t_y, t_z)
    whoami[blocks_per_grid, threads_per_block]()

if __name__ == "__main__":
    main()