# R equivalent for iterating through an array-like structure

arr <- c(12, 24, 36, 48, 60)

# Position one (first element)
cat("Position one:", arr[1], "\n")

# Iterate over the array and simulate printing memory address
for (num in arr) {
    cat(num, "\t", pryr::address(num), "\n")
    # pryr::address() returns a simulated memory address in R
}