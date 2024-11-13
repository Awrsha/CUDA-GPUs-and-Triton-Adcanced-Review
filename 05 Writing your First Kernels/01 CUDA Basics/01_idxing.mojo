def whoami():
    # Thread and block indices
    block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y
    block_offset = block_id * blockDim.x * blockDim.y * blockDim.z
    thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y

    # Global thread ID
    id = block_offset + thread_offset

    # Print the output
    print(f"{id:04d} | Block({blockIdx.x} {blockIdx.y} {blockIdx.z}) = {block_id:3d} | Thread({threadIdx.x} {threadIdx.y} {threadIdx.z}) = {thread_offset:3d}")

def main():
    b_x, b_y, b_z = 2, 3, 4
    t_x, t_y, t_z = 4, 4, 4

    blocks_per_grid = b_x * b_y * b_z
    threads_per_block = t_x * t_y * t_z

    print(f"{blocks_per_grid} blocks/grid")
    print(f"{threads_per_block} threads/block")
    print(f"{blocks_per_grid * threads_per_block} total threads")

    # Simulating the kernel launch (hypothetical in Mojo)
    launch_kernel(whoami, blocks_per_grid, threads_per_block)

if __name__ == "__main__":
    main()