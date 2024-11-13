import triton
import triton.language as tl
import torch

# Define a Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(
    A, B, C, M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Step 1: Determine the program's position within the grid along the M dimension
    # Each program will compute a sub-block of the output matrix C
    pid = tl.program_id(0)

    # Step 2: Define the offset for loading elements from matrix A and B
    # offs_m: Rows of A to be loaded
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_n: Columns of B to be loaded
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    # offs_k: Shared dimension between A and B to iterate over
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Step 3: Load sub-blocks of A and B matrices based on computed offsets
    # A_block holds a submatrix of A corresponding to the current block
    A_block = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
    # B_block holds a submatrix of B for the current block
    B_block = tl.load(B + offs_k[:, None] * N + offs_n[None, :])

    # Step 4: Initialize the accumulation matrix with zeros
    # This matrix will hold the dot-product results for the sub-block in C
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    # Perform the matrix multiplication for this block
    acc += tl.dot(A_block, B_block)

    # Step 5: Store the accumulated result back to the output matrix C
    # The calculated sub-block of C is written at the correct offset
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)

# Define input matrices and output matrix in CUDA device memory
A = torch.randn(128, 128, device="cuda")  # Random 128x128 matrix for input A
B = torch.randn(128, 128, device="cuda")  # Random 128x128 matrix for input B
C = torch.empty((128, 128), device="cuda")  # Empty matrix to store result C

# Launch the Triton kernel for matrix multiplication
# (128,) specifies the grid size, where each program computes a sub-block of the output
matmul_kernel[(128,)](
    A, B, C, 128, 128, 128, 
    BLOCK_SIZE_M=64, 
    BLOCK_SIZE_N=64, 
    BLOCK_SIZE_K=64
)

# Display the resulting matrix after matrix multiplication
print("Result:", C)