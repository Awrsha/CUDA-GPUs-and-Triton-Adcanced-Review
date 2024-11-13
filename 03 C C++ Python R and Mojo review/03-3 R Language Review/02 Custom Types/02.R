# Define a Point as a list with two float values
Point <- list(x = 1.1, y = 2.5)

# Calculate size of Point
size <- object.size(Point)
cat("Size of Point:", size, "bytes\n")  # Output is typically larger than in C, as R has overhead