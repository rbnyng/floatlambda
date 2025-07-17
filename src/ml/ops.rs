// src/ml/ops.rs

use std::rc::Rc;

use super::tensor::{Context, DifferentiableTensor};

// --- Helper for transposing a matrix ---
fn transpose(shape: &[usize], data: &[f64]) -> (Vec<usize>, Vec<f64>) {
    if shape.len() != 2 {
        panic!("Transpose only supported for 2D tensors (matrices)");
    }
    let (rows, cols) = (shape[0], shape[1]);
    let mut new_data = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            new_data[c * rows + r] = data[r * cols + c];
        }
    }
    (vec![cols, rows], new_data)
}

// --- Public Operations ---

pub fn add(t1_id: u64, t1: &DifferentiableTensor, t2_id: u64, t2: &DifferentiableTensor) -> DifferentiableTensor {
    if t1.shape != t2.shape {
        panic!("Shape mismatch for add operation: {:?} vs {:?}", t1.shape, t2.shape);
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
    result
}

pub fn matmul(t1_id: u64, t1: &DifferentiableTensor, t2_id: u64, t2: &DifferentiableTensor) -> DifferentiableTensor {
    if t1.shape.len() != 2 || t2.shape.len() != 2 || t1.shape[1] != t2.shape[0] {
        panic!("Shape mismatch for matmul: {:?} vs {:?}", t1.shape, t2.shape);
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
    let mut result = DifferentiableTensor::new(new_shape, new_data);

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

            let (t2_transposed_shape, t2_transposed_data) = transpose(&t2_shape, &t2_data);
            let grad_a_op = matmul(u64::MAX, &DifferentiableTensor::new(vec![m, n], grad_output.to_vec()),
                                   u64::MAX, &DifferentiableTensor::new(t2_transposed_shape, t2_transposed_data));
            let t1_ref = heap.get_tensor_mut(t1_id).unwrap();
            for (g_out, g_in) in grad_a_op.data.iter().zip(t1_ref.grad.borrow_mut().iter_mut()) {
                *g_in += g_out;
            }

            let (t1_transposed_shape, t1_transposed_data) = transpose(&t1_shape, &t1_data);
            let grad_b_op = matmul(u64::MAX, &DifferentiableTensor::new(t1_transposed_shape, t1_transposed_data),
                                   u64::MAX, &DifferentiableTensor::new(vec![m, n], grad_output.to_vec()));
            let t2_ref = heap.get_tensor_mut(t2_id).unwrap();
            for (g_out, g_in) in grad_b_op.data.iter().zip(t2_ref.grad.borrow_mut().iter_mut()) {
                *g_in += g_out;
            }
        }),
    }));
    result
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ops_add() {
        let t1 = DifferentiableTensor::new(vec![2], vec![10.0, 20.0]);
        let t2 = DifferentiableTensor::new(vec![2], vec![1.0, 2.0]);
        let result = add(0, &t1, 1, &t2); // IDs are dummies for this test

        assert_eq!(result.shape, vec![2]);
        assert_eq!(result.data, vec![11.0, 22.0]);
        assert!(result.context.is_some());
    }

    #[test]
    fn test_ops_matmul() {
        let t1 = DifferentiableTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let t2 = DifferentiableTensor::new(vec![2, 1], vec![5.0, 6.0]);
        let result = matmul(0, &t1, 1, &t2);

        // [1, 2] * [5] = [1*5 + 2*6] = [17]
        // [3, 4]   [6]   [3*5 + 4*6] = [39]
        assert_eq!(result.shape, vec![2, 1]);
        assert_eq!(result.data, vec![17.0, 39.0]);
        assert!(result.context.is_some());
    }

    #[test]
    #[should_panic]
    fn test_ops_matmul_bad_shapes() {
        let t1 = DifferentiableTensor::new(vec![2, 3], vec![0.0; 6]);
        let t2 = DifferentiableTensor::new(vec![2, 2], vec![0.0; 4]);
        matmul(0, &t1, 1, &t2); // Inner dimensions (3 vs 2) don't match
    }

    #[test]
    fn test_ops_sigmoid() {
        let t = DifferentiableTensor::new(vec![2], vec![0.0, 100.0]);
        let result = sigmoid(0, &t);
        
        assert!((result.data[0] - 0.5).abs() < 1e-6);
        assert!((result.data[1] - 1.0).abs() < 1e-6);
        assert!(result.context.is_some());
    }
}