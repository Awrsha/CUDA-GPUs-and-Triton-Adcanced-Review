import jax.numpy as jnp
from jax import grad, jit
import jax

# Define a simple linear model with parameters
def linear_model(params, x):
    # Perform a dot product between input features (x) and model parameters (params)
    return jnp.dot(x, params)

# Define the Mean Squared Error (MSE) loss function
def mse_loss(params, x, y):
    # Get model predictions for given inputs
    pred = linear_model(params, x)
    # Calculate and return the mean squared error between predictions and true labels (y)
    return jnp.mean((pred - y) ** 2)

# Optimization function that updates model parameters using gradient descent
@jit  # JIT compilation for faster execution
def update(params, x, y, lr=0.01):
    # Compute the gradient of the loss with respect to the parameters
    grads = grad(mse_loss)(params, x, y)
    # Update parameters using the gradient and learning rate (lr)
    return params - lr * grads

# Sample input data (features) and target labels
x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Input data
y = jnp.array([1.0, 2.0, 3.0])  # Corresponding target values

# Initialize model parameters
params = jnp.array([0.1, 0.1])  # Initial guess for weights

# Training loop
for epoch in range(100):
    # Update parameters for the current epoch
    params = update(params, x, y)
    # Print the loss every 10 epochs for monitoring progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {mse_loss(params, x, y)}")

# Display the final learned parameters
print("Final Parameters:", params)