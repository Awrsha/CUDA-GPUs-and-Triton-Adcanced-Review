# In Python, variables are references to objects, so no explicit pointers are required

x = 10
ptr = x  # 'ptr' references the same value as 'x'

# Python doesn't expose memory addresses directly, but we can use id() to get the memory address.
print(f"Address of x: {id(ptr)}")  # Prints the memory address (not exactly like C)
print(f"Value of x: {ptr}")  # Prints the value of x