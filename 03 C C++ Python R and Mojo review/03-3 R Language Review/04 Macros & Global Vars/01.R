PI <- 3.14159

# Function to calculate area
area <- function(r) {
  return(PI * r * r)
}

# Set radius with conditional logic
radius <- 7  # Default value

if (radius > 10) {
  radius <- 10
} else if (radius < 5) {
  radius <- 5
} else {
  radius <- 7
}

# Calculate and print the area
cat(sprintf("Area of circle with radius %d: %f\n", radius, area(radius)))