num <- 10
fnum <- 3.14

# Simulating the void pointer with a list containing references
vptr <- list(type = "int", value = num)
cat("Integer:", vptr$value, "\n")  # Output: 10

vptr <- list(type = "float", value = fnum)
cat("Float:", sprintf("%.2f", vptr$value), "\n")  # Output: 3.14