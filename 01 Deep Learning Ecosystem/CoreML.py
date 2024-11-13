import torch
import coremltools as ct

# Define a simple model (e.g., a PyTorch model)
class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x * 2

model = SimpleModel()

# Convert the PyTorch model to CoreML format
example_input = torch.tensor([2.0])
coreml_model = ct.convert(model, inputs=[ct.TensorType(shape=example_input.shape)])

# Save the CoreML model
coreml_model.save("simple_model.mlmodel")

print("CoreML Model has been saved as 'simple_model.mlmodel'.")
