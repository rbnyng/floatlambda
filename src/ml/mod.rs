// src/ml/mod.rs

// Declare the submodules
pub mod autodiff;
pub mod ops;
pub mod tensor;

// Re-export key components for easier use in the evaluator
pub use autodiff::grad;
pub use ops::{add, matmul, sigmoid};
pub use tensor::DifferentiableTensor;

use crate::error::EvalError;
use crate::evaluator::{list_to_vec, vec_to_list}; // We need these helpers
use crate::memory::{decode_heap_pointer, encode_heap_pointer, Heap, HeapObject};

pub fn get_ml_builtin_arity(op: &str) -> Option<usize> {
    match op {
        // Unary
        "get_data" | "get_shape" | "get_grad" |
        "transpose" | "sum_t" | "mean_t" | "sigmoid_t"
        => Some(1),

        // Binary
        "tensor" | "add_t" | "matmul" | "grad" | "reshape"
        => Some(2),

        _ => None,
    }
}

/// Executes an ML builtin operation.
/// This is the single entry point for the evaluator to call into the ML library.
pub fn execute_ml_builtin(op: &str, args: &[f64], heap: &mut Heap) -> Result<f64, EvalError> {
    match op {
        "tensor" => {
            let shape_vec = list_to_vec(args[0], heap)?.iter().map(|&x| x as usize).collect();
            let data_vec = list_to_vec(args[1], heap)?;
            let tensor = DifferentiableTensor::new(shape_vec, data_vec);
            let id = heap.register(HeapObject::Tensor(tensor));
            Ok(encode_heap_pointer(id))
        }
        "add_t" => {
            let id1 = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("add_t expects a tensor".to_string()))?;
            let id2 = decode_heap_pointer(args[1]).ok_or_else(|| EvalError::TypeError("add_t expects a tensor".to_string()))?;
            let (t1, t2) = (heap.get_tensor_mut(id1)?.clone(), heap.get_tensor_mut(id2)?.clone());
            let res = ops::add(id1, &t1, id2, &t2)?;
            let id = heap.register(HeapObject::Tensor(res));
            Ok(encode_heap_pointer(id))
        }
        "matmul" => {
            let id1 = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("matmul expects a tensor".to_string()))?;
            let id2 = decode_heap_pointer(args[1]).ok_or_else(|| EvalError::TypeError("matmul expects a tensor".to_string()))?;
            let (t1, t2) = (heap.get_tensor_mut(id1)?.clone(), heap.get_tensor_mut(id2)?.clone());
            let res = ops::matmul(id1, &t1, id2, &t2)?;
            let id = heap.register(HeapObject::Tensor(res));
            Ok(encode_heap_pointer(id))
        }
        "sigmoid_t" => {
            let id1 = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("sigmoid_t expects a tensor".to_string()))?;
            let t1 = heap.get_tensor_mut(id1)?.clone();
            let res = ops::sigmoid(id1, &t1);
            let id = heap.register(HeapObject::Tensor(res));
            Ok(encode_heap_pointer(id))
        }
        "grad" => {
            let func_ptr = args[0];
            let input_tensor_id = decode_heap_pointer(args[1]).ok_or_else(|| EvalError::TypeError("grad expects a tensor as the second argument".to_string()))?;
            let grad_tensor = autodiff::grad(func_ptr, input_tensor_id, heap)?;
            let id = heap.register(HeapObject::Tensor(grad_tensor));
            Ok(encode_heap_pointer(id))
        }
        "reshape" => {
            let id = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("reshape expects a tensor".to_string()))?;
            let new_shape_vec = list_to_vec(args[1], heap)?.iter().map(|&x| x as usize).collect();
            let tensor = heap.get_tensor_mut(id)?.clone();
            let result_tensor = ops::reshape(id, &tensor, new_shape_vec)?;
            let result_id = heap.register(HeapObject::Tensor(result_tensor));
            Ok(encode_heap_pointer(result_id))
        }
        "transpose" => {
            let id = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("transpose expects a tensor".to_string()))?;
            let tensor = heap.get_tensor_mut(id)?.clone();
            let (new_shape, new_data) = ops::transpose(&tensor.shape, &tensor.data)?;
            let result_tensor = DifferentiableTensor::new(new_shape, new_data);
            let result_id = heap.register(HeapObject::Tensor(result_tensor));
            Ok(encode_heap_pointer(result_id))
        }
        "sum_t" | "mean_t" => {
            let id = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("Operation expects a tensor".to_string()))?;
            let tensor = heap.get_tensor_mut(id)?.clone();
            let result_tensor = match op {
                "sum_t" => ops::sum_t(id, &tensor),
                "mean_t" => ops::mean_t(id, &tensor),
                _ => unreachable!(),
            };
            let result_id = heap.register(HeapObject::Tensor(result_tensor));
            Ok(encode_heap_pointer(result_id))
        }
        "get_data" | "get_shape" | "get_grad" => {
            let id = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("This operation expects a tensor".to_string()))?;
            let tensor = heap.get_tensor_mut(id)?;
            let data_to_return = match op {
                "get_data" => tensor.data.clone(),
                "get_shape" => tensor.shape.iter().map(|&x| x as f64).collect(),
                "get_grad" => tensor.grad.borrow().clone(),
                _ => unreachable!(),
            };
            Ok(vec_to_list(&data_to_return, heap))
        }
        _ => Err(EvalError::TypeError(format!("Unknown ML builtin: {}", op))),
    }
}