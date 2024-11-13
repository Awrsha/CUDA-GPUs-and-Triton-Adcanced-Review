num = 10
fnum = 3.14

# Simulating the void pointer and dereferencing by casting in Mojo
vptr = &num
print(f"Integer: {*(int*)vptr}")  # Output: 10

vptr = &fnum
print(f"Float: {*(float*)vptr:.2f}")  # Output: 3.14