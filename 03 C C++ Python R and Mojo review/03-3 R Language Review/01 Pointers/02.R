value <- 42
ptr1 <- list(value)  # List containing the value (simulates pointer to value)
ptr2 <- list(ptr1)   # List containing ptr1 (simulates pointer to pointer)
ptr3 <- list(ptr2)   # List containing ptr2 (simulates pointer to pointer to pointer)

# Access the value through multiple levels of references
cat("Value:", ptr3[[1]][[1]][[1]], "\n")  # Output: 42