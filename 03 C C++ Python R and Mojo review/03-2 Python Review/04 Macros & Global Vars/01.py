PI = 3.14159

# Function to calculate the area
def area(r):
    return PI * r * r

# Set radius with conditional logic
radius = 7  # Default value

if radius > 10:
    radius = 10
elif radius < 5:
    radius = 5
else:
    radius = 7

# Calculate and print the area
print(f"Area of circle with radius {radius}: {area(radius)}")