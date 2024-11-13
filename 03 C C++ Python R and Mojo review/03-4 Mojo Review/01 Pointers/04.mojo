// Simulating NULL pointer behavior in Mojo

var ptr: *int = null  // Initialize pointer to NULL
print("1. Initial ptr value: ", ptr)

if (ptr == null) {
    print("2. ptr is NULL, cannot dereference")
}

// Simulate memory allocation
ptr = alloc(int)  // Allocating memory for an integer
ptr! = 42  // Dereferencing and assigning a value
print("4. After allocation, ptr value: ", ptr)

if (ptr != null) {
    print("5. Value at ptr: ", ptr!)
}

// Simulate freeing memory (set to NULL after use)
ptr = null
print("6. After free, ptr value: ", ptr)

if (ptr == null) {
    print("7. ptr is NULL, safely avoided use after free")
}