# Import necessary libraries
import pyopencl as cl
import numpy as np

# Initialize an OpenCL environment
# Selecting the first platform available (usually corresponds to the first GPU or CPU)
platform = cl.get_platforms()[0]  # Get all available platforms, choose the first
device = platform.get_devices()[0]  # Get all devices of the platform, choose the first (GPU/CPU)
context = cl.Context([device])  # Create an OpenCL context for the selected device
queue = cl.CommandQueue(context)  # Create a command queue to manage execution

# Prepare input data for the OpenCL kernel
# Using numpy to create arrays of floating point numbers
a = np.array([2.0], dtype=np.float32)  # First input value (2.0)
b = np.array([3.0], dtype=np.float32)  # Second input value (3.0)

# Create OpenCL buffers for data transfer between host and device
# The buffers hold the input and output data on the device (GPU/CPU)
a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)

# OpenCL kernel source code for multiplying two numbers
# The kernel is a simple function that multiplies two elements of float arrays
program_src = """
__kernel void multiply(__global const float *a, __global const float *b, __global float *result) {
    result[0] = a[0] * b[0];  // Multiply the first element of a and b, store it in result
}
"""

# Create and build the OpenCL program from the source code
# The program is compiled and linked by the OpenCL driver on the selected device
program = cl.Program(context, program_src).build()

# Create a buffer to store the result of the multiplication
result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)  # Buffer to hold output data

# Execute the OpenCL kernel function
# The kernel 'multiply' is launched with the specified command queue, shape, and arguments
# Here, 'a.shape' defines how many work-items are involved (just 1 in this case)
program.multiply(queue, a.shape, None, a_buffer, b_buffer, result_buffer)

# Read the output result from the buffer
# After execution, we need to transfer the result back from the device to the host (CPU)
result = np.empty_like(a)  # Create an empty numpy array to store the result
cl.enqueue_copy(queue, result, result_buffer).wait()  # Enqueue the buffer copy operation

# Save the result to a file for later use
# The result is stored in the file "opencl_output.npy" for persistence
np.save("opencl_output.npy", result)

# Print the output to verify the result
print("OpenCL Output:", result)