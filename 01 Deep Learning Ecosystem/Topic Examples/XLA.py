import jax
import jax.numpy as jnp

# Example function
def f(x):
    return jnp.dot(x, x)

# Compile with XLA for optimization
xla_f = jax.jit(f)  # JIT compile the function using XLA

# Run the function with some input
x = jnp.ones((1000, 1000))
result = xla_f(x)
print(result)