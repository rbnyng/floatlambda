// src/ml/ops.rs

use std::rc::Rc;

use super::tensor::{Context, DifferentiableTensor};
use crate::error::EvalError;

// --- Helper for transposing a matrix ---
pub fn transpose(shape: &[usize], data: &[f64]) -> Result<(Vec<usize>, Vec<f64>), EvalError> {
    if shape.len() != 2 {
        return Err(EvalError::TypeError(format!(
            "Transpose only supported for 2D tensors (matrices), but got shape with {} dimensions",
            shape.len()
        )));
    }
    let (rows, cols) = (shape[0], shape[1]);
    let mut new_data = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            new_data[c * rows + r] = data[r * cols + c];
        }
    }
    Ok((vec![cols, rows], new_data))
}

// --- Public Operations ---

pub fn add(t1_id: u64, t1: &DifferentiableTensor, t2_id: u64, t2: &DifferentiableTensor) -> Result<DifferentiableTensor, EvalError> {
    if t1.shape != t2.shape {
        return Err(EvalError::TypeError(format!(
            "Shape mismatch for add operation: {:?} vs {:?}",
            t1.shape, t2.shape
        )));
    }
    let new_data: Vec<f64> = t1.data.iter().zip(t2.data.iter()).map(|(a, b)| a + b).collect();
    let mut result = DifferentiableTensor::new(t1.shape.clone(), new_data);

    result.context = Some(Rc::new(Context {
        parents: vec![t1_id, t2_id],
        backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| { 
            // Gradient of add is 1, so just pass grad back to both parents.
            let t1_ref = heap.get_tensor_mut(t1_id).unwrap();
            for (g_out, g_in) in grad_output.iter().zip(t1_ref.grad.borrow_mut().iter_mut()) {
                *g_in += g_out;
            }
            let t2_ref = heap.get_tensor_mut(t2_id).unwrap();
            for (g_out, g_in) in grad_output.iter().zip(t2_ref.grad.borrow_mut().iter_mut()) {
                *g_in += g_out;
            }
        }),
    }));
    Ok(result)
}

pub fn matmul(t1_id: u64, t1: &DifferentiableTensor, t2_id: u64, t2: &DifferentiableTensor) -> Result<DifferentiableTensor, EvalError> {
    if t1.shape.len() != 2 || t2.shape.len() != 2 {
        return Err(EvalError::TypeError(format!(
            "Matmul only supports 2D tensors, but got shapes {:?} and {:?}",
            t1.shape, t2.shape
        )));
    }
    if t1.shape[1] != t2.shape[0] {
        return Err(EvalError::TypeError(format!(
            "Shape mismatch for matmul: inner dimensions must match, but got {:?} and {:?}",
            t1.shape, t2.shape
        )));
    }
    let (m, k1) = (t1.shape[0], t1.shape[1]);
    let (_k2, n) = (t2.shape[0], t2.shape[1]);

    let mut new_data = vec![0.0; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut sum = 0.0;
            for k in 0..k1 {
                sum += t1.data[r * k1 + k] * t2.data[k * n + c];
            }
            new_data[r * n + c] = sum;
        }
    }
    let new_shape = vec![m, n];
    let mut result = DifferentiableTensor::new(new_shape.clone(), new_data);

    let t1_shape = t1.shape.clone();
    let t1_data = t1.data.clone();
    let t2_shape = t2.shape.clone();
    let t2_data = t2.data.clone();

    result.context = Some(Rc::new(Context {
        parents: vec![t1_id, t2_id],
        backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| { 
            // Backward for matmul: A * B = C
            // grad_A = grad_C * B^T
            // grad_B = A^T * grad_C

            // .unwrap() is acceptable here because the forward pass already confirmed the shapes are 2D.
            // A panic here would indicate an internal logic error.
            let (t2_transposed_shape, t2_transposed_data) = transpose(&t2_shape, &t2_data).unwrap();
            let grad_a_op = matmul(u64::MAX, &DifferentiableTensor::new(new_shape.clone(), grad_output.to_vec()),
                                   u64::MAX, &DifferentiableTensor::new(t2_transposed_shape, t2_transposed_data)).unwrap();
            let t1_ref = heap.get_tensor_mut(t1_id).unwrap();
            for (g_out, g_in) in grad_a_op.data.iter().zip(t1_ref.grad.borrow_mut().iter_mut()) {
                *g_in += g_out;
            }

            let (t1_transposed_shape, t1_transposed_data) = transpose(&t1_shape, &t1_data).unwrap();
            let grad_b_op = matmul(u64::MAX, &DifferentiableTensor::new(t1_transposed_shape, t1_transposed_data),
                                   u64::MAX, &DifferentiableTensor::new(new_shape.clone(), grad_output.to_vec())).unwrap();
            let t2_ref = heap.get_tensor_mut(t2_id).unwrap();
            for (g_out, g_in) in grad_b_op.data.iter().zip(t2_ref.grad.borrow_mut().iter_mut()) {
                *g_in += g_out;
            }
        }),
    }));
    Ok(result)
}

pub fn sigmoid(t_id: u64, t: &DifferentiableTensor) -> DifferentiableTensor {
    let new_data: Vec<f64> = t.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    let mut result = DifferentiableTensor::new(t.shape.clone(), new_data.clone());
    
    result.context = Some(Rc::new(Context {
        parents: vec![t_id],
        backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| { 
            // grad_input = grad_output * sigmoid(x) * (1 - sigmoid(x))
            let t_ref = heap.get_tensor_mut(t_id).unwrap();
            let mut grad_input = t_ref.grad.borrow_mut();
            for i in 0..grad_output.len() {
                let s_x = new_data[i];
                grad_input[i] += grad_output[i] * s_x * (1.0 - s_x);
            }
        }),
    }));
    result
}

pub fn reshape(t_id: u64, t: &DifferentiableTensor, new_shape_vec: Vec<usize>) -> Result<DifferentiableTensor, EvalError> {
    let original_len: usize = t.shape.iter().product();
    let new_len: usize = new_shape_vec.iter().product();

    if original_len != new_len {
        return Err(EvalError::TypeError(format!(
            "Cannot reshape tensor of shape {:?} ({} elements) to {:?} ({} elements)",
            t.shape, original_len, new_shape_vec, new_len
        )));
    }
    
    let mut result = DifferentiableTensor::new(new_shape_vec, t.data.clone());

    result.context = Some(Rc::new(Context {
        parents: vec![t_id],
        backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| {
            // The gradient for reshape just flows straight through, element-wise.
            let t_ref = heap.get_tensor_mut(t_id).unwrap();
            let mut grad_input = t_ref.grad.borrow_mut();
            for i in 0..grad_output.len() {
                grad_input[i] += grad_output[i];
            }
        }),
    }));
    
    Ok(result)
}

pub fn sum_t(t_id: u64, t: &DifferentiableTensor) -> DifferentiableTensor {
    let sum_val = t.data.iter().sum();
    // The result is a scalar tensor.
    let mut result = DifferentiableTensor::new(vec![], vec![sum_val]);

    result.context = Some(Rc::new(Context {
        parents: vec![t_id],
        backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| {
            // Gradient of sum is 1. We broadcast the incoming scalar gradient to all inputs.
            let scalar_grad = grad_output[0];
            let t_ref = heap.get_tensor_mut(t_id).unwrap();
            let mut grad_input = t_ref.grad.borrow_mut();
            for g_in in grad_input.iter_mut() {
                *g_in += scalar_grad;
            }
        }),
    }));

    result
}

pub fn mean_t(t_id: u64, t: &DifferentiableTensor) -> DifferentiableTensor {
    let sum_val: f64 = t.data.iter().sum();
    let count = t.data.len() as f64;
    let mean_val = if count == 0.0 { 0.0 } else { sum_val / count };
    // The result is a scalar tensor.
    let mut result = DifferentiableTensor::new(vec![], vec![mean_val]);

    if count > 0.0 {
        result.context = Some(Rc::new(Context {
            parents: vec![t_id],
            backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| {
                // Gradient of mean is 1/N. We broadcast the scaled incoming gradient.
                let scalar_grad = grad_output[0] / count;
                let t_ref = heap.get_tensor_mut(t_id).unwrap();
                let mut grad_input = t_ref.grad.borrow_mut();
                for g_in in grad_input.iter_mut() {
                    *g_in += scalar_grad;
                }
            }),
        }));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ops_add() {
        let t1 = DifferentiableTensor::new(vec![2], vec![10.0, 20.0]);
        let t2 = DifferentiableTensor::new(vec![2], vec![1.0, 2.0]);
        let result = add(0, &t1, 1, &t2).unwrap(); // IDs are dummies for this test

        assert_eq!(result.shape, vec![2]);
        assert_eq!(result.data, vec![11.0, 22.0]);
        assert!(result.context.is_some());
    }

    #[test]
    fn test_ops_matmul() {
        let t1 = DifferentiableTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let t2 = DifferentiableTensor::new(vec![2, 1], vec![5.0, 6.0]);
        let result = matmul(0, &t1, 1, &t2).unwrap();

        // [1, 2] * [5] = [1*5 + 2*6] = [17]
        // [3, 4]   [6]   [3*5 + 4*6] = [39]
        assert_eq!(result.shape, vec![2, 1]);
        assert_eq!(result.data, vec![17.0, 39.0]);
        assert!(result.context.is_some());
    }

    #[test]
    fn test_ops_matmul_bad_shapes() {
        let t1 = DifferentiableTensor::new(vec![2, 3], vec![0.0; 6]);
        let t2 = DifferentiableTensor::new(vec![2, 2], vec![0.0; 4]);
        assert!(matmul(0, &t1, 1, &t2).is_err()); // Inner dimensions (3 vs 2) don't match
    }

    #[test]
    fn test_ops_sigmoid() {
        let t = DifferentiableTensor::new(vec![2], vec![0.0, 100.0]);
        let result = sigmoid(0, &t);
        
        assert!((result.data[0] - 0.5).abs() < 1e-6);
        assert!((result.data[1] - 1.0).abs() < 1e-6);
        assert!(result.context.is_some());
    }

    #[test]
    fn test_ops_reshape() {
        let t = DifferentiableTensor::new(vec![2, 3], (1..=6).map(|x| x as f64).collect());
        let reshaped = reshape(0, &t, vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape, vec![3, 2]);
        assert_eq!(reshaped.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(reshaped.context.is_some());
    }

    #[test]
    fn test_ops_reshape_bad_size() {
        let t = DifferentiableTensor::new(vec![2, 3], vec![0.0; 6]);
        assert!(reshape(0, &t, vec![4, 2]).is_err()); // 6 elements vs 8
    }
    
    #[test]
    fn test_ops_sum_and_mean() {
        let t = DifferentiableTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let sum_res = sum_t(0, &t);
        let mean_res = mean_t(0, &t);

        assert_eq!(sum_res.shape, vec![]);
        assert_eq!(sum_res.data, vec![10.0]);
        assert!(sum_res.context.is_some());

        assert_eq!(mean_res.shape, vec![]);
        assert_eq!(mean_res.data, vec![2.5]);
        assert!(mean_res.context.is_some());
    }
}