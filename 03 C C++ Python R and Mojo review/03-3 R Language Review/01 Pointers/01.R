x <- 10
ptr <- x  # In R, variables are references

# R doesn't have a direct way to show memory addresses, but you can use the "address" of an object 
# using the "pryr" package for low-level memory inspection
library(pryr)
cat("Address of x: ", address(ptr), "\n")  # Address function from pryr package
cat("Value of x: ", ptr, "\n")