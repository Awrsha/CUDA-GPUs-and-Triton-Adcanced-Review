from llvmlite import binding

# Initialize the LLVM execution engine
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()

# Define a simple LLVM IR function (for addition)
llvm_ir = """
define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
"""

# Create a module and parse the IR
module = binding.parse_assembly(llvm_ir)
module.verify()

# Create a target machine for code generation
target = binding.Target.from_default_triple()
target_machine = target.create_target_machine()

# Create an execution engine to run the code
engine = binding.create_mcjit_compiler(module, target_machine)

# Run the function
engine.finalize_object()
engine.run_static_constructors()

# Getting function pointer
add_ptr = engine.get_function_address("add")

import ctypes
add_func = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)(add_ptr)

# Call the function
result = add_func(5, 10)
print(f"Result of addition: {result}")