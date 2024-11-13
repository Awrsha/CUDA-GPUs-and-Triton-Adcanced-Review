// Mojo equivalent for iterating through an array-like structure

let arr = [12, 24, 36, 48, 60]

// Position one (first element)
print("Position one:", arr[0])

// Iterate over the array and simulate printing memory address
for i in 0..<arr.size {
    print(arr[i], arr.pointerOf(i))  // Simulates pointer operation
}