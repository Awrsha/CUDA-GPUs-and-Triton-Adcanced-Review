import sys

arr = [12, 24, 36, 48, 60]

# Size of array
size = len(arr)
print("Size of arr:", size)  # Output: 5

# Checking the size of an integer in bytes (Python dynamically sizes integers)
print("Size of an integer in bytes:", sys.getsizeof(1))  # Typical output: 28 bytes on a 64-bit system