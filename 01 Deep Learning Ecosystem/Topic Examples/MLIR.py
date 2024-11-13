import mlir_graphblas
from mlir_graphblas import mlir

# Create a MLIR context and module
context = mlir.Context()
module = mlir.Module(context)

# Define a tensor operation using MLIR's tensor dialect
with mlir.in_context(context):
    mlir_func = """
    func @matmul(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
      %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
      return %0 : tensor<2x2xf32>
    }
    """
    
    # Parse and add the function to the module
    module.parse(mlir_func)
    
# Print the MLIR module
print(module)