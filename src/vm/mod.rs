// src/vm/mod.rs

// Declare the modules within the `vm` crate.
pub mod chunk;
pub mod opcode;
pub mod compiler;
pub mod vm; 
pub mod function; 
pub mod closure; 
pub mod natives; 

// Re-export the key structures and functions.
pub use chunk::Chunk;
pub use opcode::OpCode;
pub use compiler::compile;
pub use vm::interpret; 
pub use function::Function; 

#[cfg(test)]
mod natives_test;
mod vm_test; 
