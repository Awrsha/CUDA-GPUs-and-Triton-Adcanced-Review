import torch
import onnx
import torch_tensorrt

# Step 1: Convert PyTorch Model to ONNX Format
# Define a simple PyTorch model
class SimpleModel(torch.nn.Module):
    def forward(self, x):
        # Define forward pass: multiply input by 2
        return x * 2

# Instantiate the model
model = SimpleModel()
# Create a dummy input with dimensions (1, 3, 224, 224) to simulate a typical input image size
dummy_input = torch.randn(1, 3, 224, 224)

# Export the PyTorch model to ONNX format
# This conversion allows the model to be compatible with frameworks that support ONNX, like TensorRT
torch.onnx.export(
    model,                      # Model to be converted
    dummy_input,                # Dummy input tensor to define input size and type
    "model.onnx",               # File name for the saved ONNX model
    verbose=True                # Verbose output to display the ONNX graph structure
)

# Step 2: Load ONNX Model in TensorRT and Prepare for Inference
import tensorrt as trt

def load_engine(onnx_file_path):
    # Initialize TensorRT Logger for logging errors and warnings
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    # Create TensorRT components: builder, network, and ONNX parser
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Read the ONNX model file and parse it into the TensorRT network
        with open(onnx_file_path, 'rb') as model:
            # Parse the model to build the network
            parser.parse(model.read())
        
        # Build a CUDA engine from the network, which will optimize it for GPU inference
        engine = builder.build_cuda_engine(network)
        return engine

# Load the ONNX model into TensorRT engine format for optimized inference
engine = load_engine("model.onnx")

# Step 3: Run Inference Using TensorRT Engine
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA driver automatically

# Prepare input data for inference
# Generate random input data with shape (1, 3, 224, 224) and data type float32
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Allocate memory on the GPU for input data
d_input = cuda.mem_alloc(input_data.nbytes)
# Transfer input data from CPU to GPU memory
cuda.memcpy_htod(d_input, input_data)

# Allocate memory on the GPU for the output data
# Output size will match the input shape (1, 3, 224, 224)
output = np.empty([1, 3, 224, 224], dtype=np.float32)
d_output = cuda.mem_alloc(output.nbytes)

# Execute inference with TensorRT
# Create an execution context to manage GPU resources and inference execution
with engine.create_execution_context() as context:
    # Bind input and output data to the engine for execution
    context.execute(bindings=[int(d_input), int(d_output)])
    # Transfer the output data from GPU to CPU memory
    cuda.memcpy_dtoh(output, d_output)

# Print the inference output result
print("Inference Output:", output)