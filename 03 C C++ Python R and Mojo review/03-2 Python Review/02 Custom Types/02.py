import sys
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

# Creating an instance of Point
p = Point(1.1, 2.5)

# Calculating memory size of the instance
size = sys.getsizeof(p.x) + sys.getsizeof(p.y)
print("Size of Point:", size)  # Output varies due to Python's memory overhead, usually larger than C