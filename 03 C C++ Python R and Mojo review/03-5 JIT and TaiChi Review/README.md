# Advanced Person Class with Numba & Taichi

This project demonstrates two high-performance ways to manage and process data in Python: one using **Numba** for just-in-time (JIT) compilation and the other using **Taichi** for high-performance parallel computing. The `Person` class is implemented using both libraries to show their capabilities in handling simple data structures and operations.

## Table of Contents
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Numba Implementation](#numba-implementation)
- [Taichi Implementation](#taichi-implementation)
- [Comparison](#comparison)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Project Overview

In this project, we define a simple class `Person` that holds two attributes:
- `age`: The age of the person.
- `name`: The name of the person.

### Numba Implementation

**Numba** is a JIT compiler for Python that translates Python functions to optimized machine code at runtime. It’s particularly useful for numerical functions and performance-critical tasks.

### Taichi Implementation

**Taichi** is a high-performance computing library designed for creating efficient and parallelized computations, especially in graphics and physical simulations. It utilizes just-in-time (JIT) compilation for efficient execution on multiple architectures (e.g., CPU, GPU).

## Dependencies

To run the project, you need to install the following dependencies:

```bash
pip install numba taichi
```

## Numba Implementation

The `Person` class is implemented using **Numba**'s `@jitclass` decorator. This allows Python objects to be compiled to efficient machine code while retaining their simplicity and structure.

### Code for Numba:

```python
from numba import int32, types, jitclass
from numba.experimental import jitclass

# Define the spec for the jitclass
spec = [
    ('age', int32),
    ('name', types.unicode_type),
]

@jitclass(spec)
class Person:
    def __init__(self, age, name):
        self.age = age
        self.name = name

# Create person1
person1 = Person(25, "elliot")
print(f"age: {person1.age}\tname: {person1.name}")

# Create person2
person2 = Person(20, "not elliot")
print(f"age: {person2.age}\tname: {person2.name}")
```

### Explanation:
- **JIT Compilation**: Numba compiles the `Person` class to machine code for faster execution.
- **Unicode String Support**: The `name` attribute is defined as `unicode_type` to handle string data.

### Output:
```plaintext
age: 25    name: elliot
age: 20    name: not elliot
```

## Taichi Implementation

Taichi provides an efficient way to handle computation-heavy tasks, especially useful for simulating and rendering physics, and is highly optimized for GPU computation. Although Taichi does not natively support strings, we treat the `name` attribute as a regular Python string.

### Code for Taichi:

```python
import taichi as ti

ti.init(arch=ti.cpu)  # Initialize Taichi with CPU backend

# Define a Person struct with Taichi fields
@ti.data_oriented
class Person:
    def __init__(self, age, name):
        self.age = ti.field(dtype=ti.i32, shape=())
        self.age[None] = age
        self.name = name  # Strings are not natively supported, so handle as-is for simplicity

    @ti.kernel
    def print(self):
        ti.print(f"age: {self.age[None]}\tname: {self.name}")

# Instantiate Person and print
person1 = Person(25, "elliot")
person1.print()

person2 = Person(20, "not elliot")
person2.print()
```

### Explanation:
- **Taichi Fields**: The `age` attribute is handled by Taichi's `field` type for efficient memory access.
- **Printing**: The `print` method uses Taichi's `ti.print` for output, designed for high-performance computing.

### Output:
```plaintext
age: 25    name: elliot
age: 20    name: not elliot
```

## Comparison

| Feature             | Numba                          | Taichi                           |
|---------------------|--------------------------------|----------------------------------|
| **Performance**      | Optimized for CPU-bound tasks  | Optimized for parallel and GPU   |
| **Ease of Use**      | Straightforward integration    | Best for simulation & physics    |
| **String Support**   | Supports unicode strings       | Limited string support           |
| **Parallelism**      | Limited parallelism            | Designed for high parallelism    |

## Usage

To run this code:
1. Clone this repository.
2. Install the dependencies (`numba` and `taichi`).
3. Run the Python script to see how each implementation handles the `Person` class.

For Numba, simply run:

```bash
python 01.py
```

For Taichi, run:

```bash
python 02.py
```

Both scripts will output the age and name of two persons.

## Acknowledgments

- **Numba**: A high-performance Python compiler from the creators of Anaconda, which optimizes numeric computation.
- **Taichi**: A high-performance, data-oriented programming framework designed for fast numerical computing.
  
## License

This project is licensed under the MIT License. See `LICENSE` for details.
