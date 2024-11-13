// Mojo equivalent for simulating matrix-like behavior using arrays

let arr1 = [1, 2, 3, 4]
let arr2 = [5, 6, 7, 8]
let matrix = [arr1, arr2]

for row in matrix {
    for value in row {
        print(value, end=" ")
    }
    print()  // Newline after each row
}