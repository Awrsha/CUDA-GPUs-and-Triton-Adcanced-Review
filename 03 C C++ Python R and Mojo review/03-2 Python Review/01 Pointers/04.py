# Simulating NULL pointer behavior in Python using None

ptr = None  # Initializing to None (similar to NULL in C)
print(f"1. Initial ptr value: {ptr}")

# Check for None (NULL) before using
if ptr is None:
    print("2. ptr is None, cannot dereference")

# Simulate memory allocation
ptr = 42  # Allocating a value to ptr
print(f"4. After allocation, ptr value: {ptr}")

# Safe to use ptr after checking for None
if ptr is not None:
    print(f"5. Value at ptr: {ptr}")

# Simulate freeing memory (set to None after use)
ptr = None
print(f"6. After free, ptr value: {ptr}")

# Safety check after 'free' (set to None)
if ptr is None:
    print("7. ptr is None, safely avoided use after free")