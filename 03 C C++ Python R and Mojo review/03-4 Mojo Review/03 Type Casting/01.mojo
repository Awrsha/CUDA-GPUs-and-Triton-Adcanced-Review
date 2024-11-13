# Mojo version

fn main():
    # c-style type casting
    f: Float32 = 69.69
    i: Int32 = Int32(f)  # Cast float to int
    print(i)  # Output: 69

    # to char
    c: Char = Char(i)  # Cast int to char
    print(c)  # Output: 'E'