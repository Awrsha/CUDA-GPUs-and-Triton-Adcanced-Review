fn area(r: Float) -> Float:
    PI = 3.14159
    return PI * r * r

fn main():
    # Set initial radius value
    radius = 7  # Default radius value

    # Conditional checks
    if radius > 10:
        radius = 10
    elif radius < 5:
        radius = 5
    else:
        radius = 7

    # Calculate and print the area
    print(f"Area of circle with radius {radius}: {area(radius)}")