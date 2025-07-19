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

pub fn relu(t_id: u64, t: &DifferentiableTensor) -> DifferentiableTensor {
    let new_data: Vec<f64> = t.data.iter().map(|&x| x.max(0.0)).collect();
    let mut result = DifferentiableTensor::new(t.shape.clone(), new_data);
    
    // The backward pass needs the original input data to determine where the gradient is zero.
    let original_data = t.data.clone();
    result.context = Some(Rc::new(Context {
        parents: vec![t_id],
        backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| {
            // grad_input = grad_output * (1 if x > 0 else 0)
            let t_ref = heap.get_tensor_mut(t_id).unwrap();
            let mut grad_input = t_ref.grad.borrow_mut();
            for i in 0..grad_output.len() {
                if original_data[i] > 0.0 {
                    grad_input[i] += grad_output[i];
                }
            }
        }),
    }));
    result
}

pub fn mse_loss(y_true_id: u64, y_true: &DifferentiableTensor, y_pred_id: u64, y_pred: &DifferentiableTensor) -> Result<DifferentiableTensor, EvalError> {
    if y_true.shape != y_pred.shape {
        return Err(EvalError::TypeError(format!(
            "Shape mismatch for mse_loss: y_true {:?} vs y_pred {:?}",
            y_true.shape, y_pred.shape
        )));
    }
    
    // Forward pass: mean((y_pred - y_true)^2)
    let n = y_true.data.len() as f64;
    let diffs: Vec<f64> = y_pred.data.iter().zip(y_true.data.iter()).map(|(p, t)| p - t).collect();
    let squared_error_sum: f64 = diffs.iter().map(|&d| d * d).sum();
    let loss = squared_error_sum / n;

    // The result is a scalar tensor.
    let mut result = DifferentiableTensor::new(vec![], vec![loss]);

    // Backward pass
    result.context = Some(Rc::new(Context {
        parents: vec![y_true_id, y_pred_id],
        backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| {
            // grad_pred = grad_output * 2 * (y_pred - y_true) / N
            // grad_true = grad_output * -2 * (y_pred - y_true) / N
            let scalar_grad = grad_output[0];
            
            // Use a separate scope for each mutable borrow.
            {
                let y_pred_ref = heap.get_tensor_mut(y_pred_id).unwrap();
                let mut grad_pred = y_pred_ref.grad.borrow_mut();
                for i in 0..diffs.len() {
                    grad_pred[i] += scalar_grad * 2.0 * diffs[i] / n;
                }
            }

            {
                let y_true_ref = heap.get_tensor_mut(y_true_id).unwrap();
                let mut grad_true = y_true_ref.grad.borrow_mut();
                for i in 0..diffs.len() {
                    grad_true[i] += scalar_grad * -2.0 * diffs[i] / n;
                }
            }
        }),
    }));
    Ok(result)
}

pub fn flatten(t_id: u64, t: &DifferentiableTensor) -> Result<DifferentiableTensor, EvalError> {
    if t.shape.is_empty() {
        return Err(EvalError::TypeError("Cannot flatten a scalar tensor.".to_string()));
    }
    if t.shape.len() == 1 { // Already flat
        return Ok(t.clone());
    }

    let batch_size = t.shape[0];
    let other_dims_product: usize = t.shape[1..].iter().product();
    let new_shape = vec![batch_size, other_dims_product];
    
    let mut result = DifferentiableTensor::new(new_shape, t.data.clone());

    result.context = Some(Rc::new(Context {
        parents: vec![t_id],
        backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| {
            // The gradient for flatten is just an un-flatten (a reshape).
            // Since the data layout is identical, we just add the gradients.
            let t_ref = heap.get_tensor_mut(t_id).unwrap();
            let mut grad_input = t_ref.grad.borrow_mut();
            assert_eq!(grad_output.len(), grad_input.len());
            for i in 0..grad_output.len() {
                grad_input[i] += grad_output[i];
            }
        }),
    }));
    Ok(result)
}

pub fn softmax_ce_loss(_y_true_id: u64, y_true: &DifferentiableTensor, logits_id: u64, logits: &DifferentiableTensor) -> Result<DifferentiableTensor, EvalError> {
    if y_true.shape != logits.shape || y_true.shape.len() != 2 {
        return Err(EvalError::TypeError("softmax_ce_loss expects 2D tensors [batch, classes] of the same shape.".to_string()));
    }
    
    let batch_size = logits.shape[0];
    let num_classes = logits.shape[1];
    let n = batch_size as f64;
    
    let mut softmax_probs = Vec::with_capacity(logits.data.len());
    let mut total_loss = 0.0;

    // Forward pass
    for i in 0..batch_size {
        let start = i * num_classes;
        let end = start + num_classes;
        let logit_slice = &logits.data[start..end];
        
        // Stabilize by subtracting max logit
        let max_logit = logit_slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f64> = logit_slice.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exps: f64 = exps.iter().sum();

        let probs: Vec<f64> = exps.iter().map(|&e| e / sum_exps).collect();
        
        // Calculate loss for this item: -sum(y_true * log(probs))
        let y_true_slice = &y_true.data[start..end];
        let item_loss: f64 = y_true_slice.iter().zip(probs.iter()).map(|(yt, p)| -yt * p.ln()).sum();
        total_loss += item_loss;
        
        softmax_probs.extend(probs);
    }

    let mean_loss = total_loss / n;
    let mut result = DifferentiableTensor::new(vec![], vec![mean_loss]);

    // Backward pass
    let y_true_data = y_true.data.clone();
    result.context = Some(Rc::new(Context {
        parents: vec![logits_id], // Loss flows back only to logits
        backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| {
            // grad_logits = grad_output * (softmax_probs - y_true) / N
            let scalar_grad = grad_output[0];
            let logits_ref = heap.get_tensor_mut(logits_id).unwrap();
            let mut grad_logits = logits_ref.grad.borrow_mut();

            for i in 0..softmax_probs.len() {
                grad_logits[i] += scalar_grad * (softmax_probs[i] - y_true_data[i]) / n;
            }
        }),
    }));

    Ok(result)
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