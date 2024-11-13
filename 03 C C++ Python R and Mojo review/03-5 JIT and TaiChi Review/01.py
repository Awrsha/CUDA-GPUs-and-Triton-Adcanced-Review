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