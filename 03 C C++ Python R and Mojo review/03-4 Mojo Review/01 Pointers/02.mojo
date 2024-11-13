value = 42
ptr1 = &value  # Pointer to value
ptr2 = &ptr1   # Pointer to pointer
ptr3 = &ptr2   # Pointer to pointer to pointer

# Access the value through multiple levels of dereferencing
print(f"Value: {***ptr3}")  # Output: 42