# Python equivalent for iterating through an array-like structure

arr = [12, 24, 36, 48, 60]

# Position one (first element)
print(f"Position one: {arr[0]}")

# Iterate over the array and simulate printing memory address
for num in arr:
    print(f"{num}\t{id(num)}")
    # id() gives a unique identifier for the object, similar to a memory address