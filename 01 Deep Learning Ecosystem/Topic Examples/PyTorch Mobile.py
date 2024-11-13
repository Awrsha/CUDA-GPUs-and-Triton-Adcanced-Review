import torch
import torch.nn as nn

# Define a simple PyTorch model
class SimpleModel(nn.Module):
    def forward(self, x):
        return x * 2

# Train or load a pre-trained model
model = SimpleModel()

# Save the model in TorchScript format (required for PyTorch Mobile)
scripted_model = torch.jit.script(model)
scripted_model.save("simple_model.pt")

# Load the model on the mobile app (pseudo-code for mobile)
# model = torch.jit.load("simple_model.pt")
# output = model(input_data)