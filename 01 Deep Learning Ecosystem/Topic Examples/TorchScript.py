import torch

# Define a simple PyTorch model
# This model multiplies the input tensor by 2
class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x * 2  # The forward method specifies how the input is processed

# Instantiate the model
model = SimpleModel()

# Convert the model to TorchScript using the torch.jit.script function
# This approach allows for converting dynamic models by scripting Python code into a TorchScript
scripted_model = torch.jit.script(model)

# Save the scripted model to a file in .pt format
# TorchScript models can be serialized for later use, making them portable
scripted_model.save("simple_model_scripted.pt")

# Load the scripted model from the saved file
# This loaded model can now be used in a production environment without the full PyTorch environment
loaded_model = torch.jit.load("simple_model_scripted.pt")

# Define an input tensor to test the model with
input_data = torch.tensor([2.0])

# Run inference using the loaded TorchScript model
# This computes the result by multiplying the input by 2 as defined in the SimpleModel
output = loaded_model(input_data)

# Print the output result
print("TorchScript Output:", output)