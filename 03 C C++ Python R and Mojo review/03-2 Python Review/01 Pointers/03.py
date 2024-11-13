# Python does not have pointers like C, but we can simulate similar behavior using references

num = 10
fnum = 3.14

# Simulating a void pointer behavior by storing references in a dictionary
vptr = {"type": "int", "value": num}
print(f"Integer: {vptr['value']}")  # Output: 10

vptr = {"type": "float", "value": fnum}
print(f"Float: {vptr['value']:.2f}")  # Output: 3.14