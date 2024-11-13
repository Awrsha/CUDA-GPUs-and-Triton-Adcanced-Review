# Import necessary libraries
import torch
import rocm  # ROCm (Radeon Open Compute) library for AMD GPUs

# Check if you are using ROCm as a backend
device = torch.device("cuda")  # Use CUDA for NVIDIA GPUs, or change to 'rocm' if using ROCm on AMD GPUs

# Define a simple model for testing purposes
class SimpleModel(torch.nn.Module):
    def forward(self, x):
        # A simple forward pass multiplying input by 2
        return x * 2

# Instantiate the model and transfer it to the correct device (either CUDA or ROCm)
model = SimpleModel().to(device)

# Define input data and move it to the same device as the model
input_data = torch.tensor([2.0]).to(device)

# Run the model with the input data and get the output
output = model(input_data)

# Save the output tensor to a file for later use (e.g., for debugging or analysis)
torch.save(output, "rocm_output.pth")

# Print the output to the console for inspection
print("ROCm Output:", output)
