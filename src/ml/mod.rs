// src/ml/mod.rs

// Declare the submodules
pub mod autodiff;
pub mod ops;
pub mod tensor;

// Re-export key components for easier use in the evaluator
pub use autodiff::grad;
pub use ops::{add, matmul, sigmoid};
pub use tensor::DifferentiableTensor;