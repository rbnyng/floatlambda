// src/vm/mod.rs

// Declare the modules within the `vm` crate.
pub mod chunk;
pub mod opcode;
pub mod compiler;
pub mod vm; // New
pub mod vm_test; // New
pub mod function; // New
pub mod closure; // New

// Re-export the key structures and functions.
pub use chunk::Chunk;
pub use opcode::OpCode;
pub use compiler::compile;
pub use vm::interpret; // New
pub use function::Function; // New
