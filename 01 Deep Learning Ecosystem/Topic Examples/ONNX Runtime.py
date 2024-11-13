import torch
import onnx
import onnxruntime as ort

# Define a simple PyTorch model
# This model performs a simple operation of multiplying the input tensor by 2.
class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x * 2  # Forward pass: multiplies the input by 2.

# Instantiate the PyTorch model
model = SimpleModel()

# Convert the PyTorch model to ONNX format
# ONNX (Open Neural Network Exchange) is an open format used for transferring models between frameworks.
# The dummy input tensor (of shape [1, 3, 224, 224]) is used as a placeholder for the model's input.
dummy_input = torch.randn(1, 3, 224, 224)  # Dummy input for the conversion
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)
# `opset_version=11` ensures compatibility with ONNX opsets for export.

# Load the ONNX model using ONNX Runtime
# ONNX Runtime is an optimized engine for running ONNX models across different platforms.
ort_session = ort.InferenceSession("model.onnx")
# `InferenceSession` is used to load and run the ONNX model.

# Prepare input data for inference
input_data = dummy_input.numpy()  # Convert the PyTorch tensor to a NumPy array for ONNX Runtime
ort_inputs = {ort_session.get_inputs()[0].name: input_data}
# `ort_session.get_inputs()[0].name` retrieves the name of the model’s input layer.

# Perform inference with ONNX Runtime
ort_outs = ort_session.run(None, ort_inputs)
# `ort_session.run(None, ort_inputs)` performs the inference. The `None` argument means return all outputs.

# Print the result of the inference
print("ONNX Runtime Output:", ort_outs[0])