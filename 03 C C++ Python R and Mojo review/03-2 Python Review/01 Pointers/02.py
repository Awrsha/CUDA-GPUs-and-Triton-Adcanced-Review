value = 42
ptr1 = [value]  # List containing the value (simulates pointer to value)
ptr2 = [ptr1]   # List containing ptr1 (simulates pointer to pointer)
ptr3 = [ptr2]   # List containing ptr2 (simulates pointer to pointer to pointer)

# Access the value through multiple levels of references
print(f"Value: {ptr3[0][0][0]}")  # Output: 42