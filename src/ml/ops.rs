// src/ml/ops.rs

use std::rc::Rc;
use super::tensor::{Context, DifferentiableTensor};
use crate::error::EvalError;
use crate::memory::Heap;

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
    
    let original_data = t.data.clone();
    result.context = Some(Rc::new(Context {
        parents: vec![t_id],
        backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| {
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
    
    let n = y_true.data.len() as f64;
    let diffs: Vec<f64> = y_pred.data.iter().zip(y_true.data.iter()).map(|(p, t)| p - t).collect();
    let squared_error_sum: f64 = diffs.iter().map(|&d| d * d).sum();
    let loss = squared_error_sum / n;

    let mut result = DifferentiableTensor::new(vec![], vec![loss]);

    result.context = Some(Rc::new(Context {
        parents: vec![y_true_id, y_pred_id],
        backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| {
            let scalar_grad = grad_output[0];
            
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

pub fn conv2d(
    input_id: u64, input: &DifferentiableTensor,
    weights_id: u64, weights: &DifferentiableTensor,
    bias_id: u64, bias: &DifferentiableTensor,
    stride: usize, padding: usize
) -> Result<DifferentiableTensor, EvalError> {
    // Expecting NHWC layout: [Batch, Height, Width, Channels]
    // Weights: [KernelH, KernelW, InChannels, OutChannels]
    // Bias: [OutChannels]
    if input.shape.len() != 4 || weights.shape.len() != 4 || bias.shape.len() != 1 {
        return Err(EvalError::TypeError("Invalid tensor dimensions for conv2d".to_string()));
    }

    let (n, in_h, in_w, in_c) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    let (kh, kw, _, out_c) = (weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]);
    if in_c != weights.shape[2] || out_c != bias.shape[0] {
        return Err(EvalError::TypeError("Channel mismatch in conv2d".to_string()));
    }

    let out_h = (in_h + 2 * padding - kh) / stride + 1;
    let out_w = (in_w + 2 * padding - kw) / stride + 1;

    let mut out_data = vec![0.0; n * out_h * out_w * out_c];

    // Forward Pass
    for i in 0..n { // Batch
        for y in 0..out_h { // Output Height
            for x in 0..out_w { // Output Width
                for k in 0..out_c { // Output Channel
                    let mut sum = 0.0;
                    for ky in 0..kh { // Kernel Height
                        for kx in 0..kw { // Kernel Width
                            for c in 0..in_c { // Input Channel
                                let in_y = (y * stride + ky) as isize - padding as isize;
                                let in_x = (x * stride + kx) as isize - padding as isize;

                                if in_y >= 0 && in_y < in_h as isize && in_x >= 0 && in_x < in_w as isize {
                                    let in_idx = i * (in_h * in_w * in_c) + (in_y as usize * in_w * in_c) + (in_x as usize * in_c) + c;
                                    let w_idx = ky * (kw * in_c * out_c) + kx * (in_c * out_c) + c * out_c + k;
                                    sum += input.data[in_idx] * weights.data[w_idx];
                                }
                            }
                        }
                    }
                    let out_idx = i * (out_h * out_w * out_c) + y * (out_w * out_c) + x * out_c + k;
                    out_data[out_idx] = sum + bias.data[k];
                }
            }
        }
    }
    
    let mut result = DifferentiableTensor::new(vec![n, out_h, out_w, out_c], out_data);
    let input_data = input.data.clone();
    let weights_data = weights.data.clone();
    
    // Extract the length of the bias vector here, before the closure.
    let bias_len = bias.data.len();
    
    result.context = Some(Rc::new(Context {
        parents: vec![input_id, weights_id, bias_id],
        backward_fn: Box::new(move |grad_output, heap: &mut Heap| {
            // Create temporary vectors to store calculated gradients.
            let mut temp_grad_input = vec![0.0; input_data.len()];
            let mut temp_grad_weights = vec![0.0; weights_data.len()];
            let mut temp_grad_bias = vec![0.0; bias_len];

            // Perform all calculations on these temporary vectors.
            for i in 0..n {
                for y in 0..out_h {
                    for x in 0..out_w {
                        for k in 0..out_c {
                            let out_idx = i * (out_h * out_w * out_c) + y * (out_w * out_c) + x * out_c + k;
                            let grad_out_val = grad_output[out_idx];
                            
                            temp_grad_bias[k] += grad_out_val;
                            
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    for c in 0..in_c {
                                        let in_y = (y * stride + ky) as isize - padding as isize;
                                        let in_x = (x * stride + kx) as isize - padding as isize;

                                        if in_y >= 0 && in_y < in_h as isize && in_x >= 0 && in_x < in_w as isize {
                                            let in_idx = i * (in_h * in_w * in_c) + (in_y as usize * in_w * in_c) + (in_x as usize * in_c) + c;
                                            let w_idx = ky * (kw * in_c * out_c) + kx * (in_c * out_c) + c * out_c + k;
                                            
                                            temp_grad_weights[w_idx] += input_data[in_idx] * grad_out_val;
                                            temp_grad_input[in_idx] += weights_data[w_idx] * grad_out_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // After calculations are done, apply the results to the real tensors
            // in sequential, non-overlapping mutable borrows.
            {
                let mut grad_input = heap.get_tensor_mut(input_id).unwrap().grad.borrow_mut();
                for (g, t) in grad_input.iter_mut().zip(temp_grad_input.iter()) { *g += t; }
            }
            {
                let mut grad_weights = heap.get_tensor_mut(weights_id).unwrap().grad.borrow_mut();
                for (g, t) in grad_weights.iter_mut().zip(temp_grad_weights.iter()) { *g += t; }
            }
            {
                let mut grad_bias = heap.get_tensor_mut(bias_id).unwrap().grad.borrow_mut();
                for (g, t) in grad_bias.iter_mut().zip(temp_grad_bias.iter()) { *g += t; }
            }
        }),
    }));
    Ok(result)
}

pub fn max_pool2d(input_id: u64, input: &DifferentiableTensor, kernel_size: usize, stride: usize) -> Result<DifferentiableTensor, EvalError> {
    if input.shape.len() != 4 {
        return Err(EvalError::TypeError("max_pool2d expects a 4D tensor [N, H, W, C]".to_string()));
    }
    let (n, in_h, in_w, c) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);

    let out_h = (in_h - kernel_size) / stride + 1;
    let out_w = (in_w - kernel_size) / stride + 1;

    let mut out_data = vec![0.0; n * out_h * out_w * c];
    let mut switches = vec![0; n * out_h * out_w * c]; // To store indices of max values

    // Forward Pass
    for i in 0..n {
        for ch in 0..c {
            for y in 0..out_h {
                for x in 0..out_w {
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_idx = 0;
                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let in_y = y * stride + ky;
                            let in_x = x * stride + kx;
                            let in_idx = i * (in_h * in_w * c) + in_y * (in_w * c) + in_x * c + ch;
                            if input.data[in_idx] > max_val {
                                max_val = input.data[in_idx];
                                max_idx = in_idx;
                            }
                        }
                    }
                    let out_idx = i * (out_h * out_w * c) + y * (out_w * c) + x * c + ch;
                    out_data[out_idx] = max_val;
                    switches[out_idx] = max_idx;
                }
            }
        }
    }
    
    let mut result = DifferentiableTensor::new(vec![n, out_h, out_w, c], out_data);
    result.context = Some(Rc::new(Context {
        parents: vec![input_id],
        backward_fn: Box::new(move |grad_output, heap: &mut Heap| {
            // Backward Pass
            let mut grad_input = heap.get_tensor_mut(input_id).unwrap().grad.borrow_mut();
            for (i, &grad) in grad_output.iter().enumerate() {
                let max_idx = switches[i];
                grad_input[max_idx] += grad;
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
        
        let max_logit = logit_slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f64> = logit_slice.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exps: f64 = exps.iter().sum();

        let probs: Vec<f64> = exps.iter().map(|&e| e / sum_exps).collect();
        
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
        parents: vec![logits_id],
        backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| {
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
    let mut result = DifferentiableTensor::new(vec![], vec![sum_val]);
    result.context = Some(Rc::new(Context {
        parents: vec![t_id],
        backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| {
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
    let mut result = DifferentiableTensor::new(vec![], vec![mean_val]);
    if count > 0.0 {
        result.context = Some(Rc::new(Context {
            parents: vec![t_id],
            backward_fn: Box::new(move |grad_output, heap: &mut crate::memory::Heap| {
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