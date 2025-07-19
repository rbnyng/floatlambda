// src/ml/tensor.rs

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::memory::Heap;

// The context for backpropagation. It stores the tensors that were inputs
// to the operation and a function that can compute the backward pass.
pub struct Context {
    pub parents: Vec<u64>,
    pub backward_fn: Box<dyn Fn(&[f64], &mut Heap)>,
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("parents", &self.parents)
            .field("backward_fn", &"<closure>")
            .finish()
    }
}

// The core Tensor object for the ML library. It supports automatic differentiation.
#[derive(Debug, Clone)]
pub struct DifferentiableTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    // The gradient, wrapped in a RefCell for interior mutability during backprop.
    pub grad: RefCell<Vec<f64>>,
    // The operation that created this tensor, if any.
    pub context: Option<Rc<Context>>,
}

impl Clone for Context {
    fn clone(&self) -> Self {
        panic!("Context objects are not meant to be cloned directly");
    }
}

impl DifferentiableTensor {
    // Creates a new tensor.
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Self {
        let grad_data = vec![0.0; data.len()];
        DifferentiableTensor {
            shape,
            data,
            grad: RefCell::new(grad_data),
            context: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_new() {
        let shape = vec![2, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = DifferentiableTensor::new(shape.clone(), data.clone());

        assert_eq!(t.shape, shape);
        assert_eq!(t.data, data);
        assert_eq!(*t.grad.borrow(), vec![0.0, 0.0, 0.0, 0.0]);
        assert!(t.context.is_none());
    }
}