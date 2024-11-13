struct Point:
    var x: Float32
    var y: Float32

# Create an instance of Point
let p = Point(1.1, 2.5)

# Calculate the size of the struct
print("Size of Point:", sizeof(Point))  // Expected output: 8 bytes (4 bytes for each float)