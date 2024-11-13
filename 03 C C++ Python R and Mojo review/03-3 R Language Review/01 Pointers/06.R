# R equivalent for simulating matrix-like behavior using lists

arr1 <- c(1, 2, 3, 4)
arr2 <- c(5, 6, 7, 8)
matrix <- list(arr1, arr2)

for (row in matrix) {
  for (value in row) {
    cat(value, " ")
  }
  cat("\n")
}