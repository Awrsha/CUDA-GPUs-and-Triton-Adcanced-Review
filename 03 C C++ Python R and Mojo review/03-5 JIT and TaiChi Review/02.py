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