# Simulating NULL pointer behavior in R

ptr <- NULL  # Initialize ptr as NULL
cat("1. Initial ptr value:", ptr, "\n")

# Check for NULL before using
if (is.null(ptr)) {
    cat("2. ptr is NULL, cannot dereference\n")
}

# Simulate memory allocation
ptr <- 42  # Assigning a value to ptr (simulating allocation)
cat("4. After allocation, ptr value:", ptr, "\n")

# Safe to use ptr after NULL check
if (!is.null(ptr)) {
    cat("5. Value at ptr:", ptr, "\n")
}

# Simulate freeing memory (set to NULL after use)
ptr <- NULL
cat("6. After free, ptr value:", ptr, "\n")

# Safety check after 'free' (set to NULL)
if (is.null(ptr)) {
    cat("7. ptr is NULL, safely avoided use after free\n")
}