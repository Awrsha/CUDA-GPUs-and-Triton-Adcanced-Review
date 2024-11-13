# Python equivalent for simulating matrix-like behavior using lists

arr1 = [1, 2, 3, 4]
arr2 = [5, 6, 7, 8]
matrix = [arr1, arr2]

for row in matrix:
    for i in range(len(row)):
        print(row[i], end=" ")
    print()