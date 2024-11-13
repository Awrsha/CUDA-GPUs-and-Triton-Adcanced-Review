library(gpuR)

# Kernel function to simulate the CUDA functionality
whoami_kernel <- function() {
  block_id <- threadIdx[1] + (blockIdx[1] - 1) * blockDim[1]
  block_offset <- block_id * (blockDim[1] * blockDim[2] * blockDim[3])
  thread_offset <- threadIdx[1] + threadIdx[2] * blockDim[1] + threadIdx[3] * blockDim[1] * blockDim[2]
  id <- block_offset + thread_offset
  
  # Print the global thread ID
  cat(sprintf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n", id, blockIdx[1], blockIdx[2], blockIdx[3], block_id, threadIdx[1], threadIdx[2], threadIdx[3], thread_offset))
}

# Simulate the grid and thread launch
blocks_per_grid <- c(2, 3, 4)
threads_per_block <- c(4, 4, 4)

# Call the kernel function
whoami_kernel()