import torch

# Define a simple PyTorch model
# This model takes an input tensor and returns the input multiplied by 2
class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x * 2  # Forward pass defines a simple operation of doubling the input

# Instantiate the model
model = SimpleModel()

# Compile the model using torch.compile (available in PyTorch 2.0+)
# torch.compile optimizes the model for faster runtime, making it ideal for production environments
compiled_model = torch.compile(model)

# Define input data for inference
input_data = torch.tensor([2.0])

# Run inference with the compiled model
# The compiled model should give the same result as the original model but potentially with improved performance
output = compiled_model(input_data)

# Print the output result
print("torch.compile Output:", output)