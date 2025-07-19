// src/ml/mod.rs

// Declare the submodules
pub mod autodiff;
pub mod ops;
pub mod tensor;
pub mod optimizers;

// Re-export key components for easier use in the evaluator
pub use autodiff::grad;
pub use ops::{add, matmul, sigmoid};
pub use tensor::DifferentiableTensor;
use crate::error::EvalError;
use crate::interpreter::evaluator::{list_to_vec, vec_to_list};
use crate::memory::{decode_heap_pointer, encode_heap_pointer, Heap, HeapObject, NIL_VALUE};

pub fn get_ml_builtin_arity(op: &str) -> Option<usize> {
    match op {
        // Unary
        "get_data" | "get_shape" | "get_grad" |
        "transpose" | "sum_t" | "mean_t" |
        "sigmoid_t" | "relu_t" |
        "adamw_init_state" |
        "flatten"
        => Some(1),

        // Binary
        "tensor" | "add_t" | "matmul" | "grad" | "reshape" |
        "mse_loss" | "softmax_ce_loss"
        => Some(2),

        // Ternary
        "sgd_update" |
        "max_pool2d" // --- ADDED ---
        => Some(3),

        // 8 arguments
        "adamw_update"
        => Some(8),

        // 5 arguments: input, weights, bias, stride, padding
        "conv2d"
        => Some(5),

        _ => None,
    }
}

// Executes an ML builtin operation.
pub fn execute_ml_builtin(op: &str, args: &[f64], heap: &mut Heap) -> Result<f64, EvalError> {
    match op {
        "conv2d" => {
            let input_id = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("conv2d expects a tensor for input".to_string()))?;
            let weights_id = decode_heap_pointer(args[1]).ok_or_else(|| EvalError::TypeError("conv2d expects a tensor for weights".to_string()))?;
            let bias_id = decode_heap_pointer(args[2]).ok_or_else(|| EvalError::TypeError("conv2d expects a tensor for bias".to_string()))?;
            let stride = args[3] as usize;
            let padding = args[4] as usize;

            let input = heap.get_tensor_mut(input_id)?.clone();
            let weights = heap.get_tensor_mut(weights_id)?.clone();
            let bias = heap.get_tensor_mut(bias_id)?.clone();

            let res = ops::conv2d(input_id, &input, weights_id, &weights, bias_id, &bias, stride, padding)?;
            let res_id = heap.register(HeapObject::Tensor(res));
            Ok(encode_heap_pointer(res_id))
        }
        "max_pool2d" => {
            let input_id = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("max_pool2d expects a tensor for input".to_string()))?;
            let kernel_size = args[1] as usize;
            let stride = args[2] as usize;
            
            let input = heap.get_tensor_mut(input_id)?.clone();

            let res = ops::max_pool2d(input_id, &input, kernel_size, stride)?;
            let res_id = heap.register(HeapObject::Tensor(res));
            Ok(encode_heap_pointer(res_id))
        }
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
        "relu_t" => {
            let id = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("relu_t expects a tensor".to_string()))?;
            let t = heap.get_tensor_mut(id)?.clone();
            let res = ops::relu(id, &t);
            let res_id = heap.register(HeapObject::Tensor(res));
            Ok(encode_heap_pointer(res_id))
        }
        "mse_loss" => {
            let y_true_id = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("mse_loss expects a tensor".to_string()))?;
            let y_pred_id = decode_heap_pointer(args[1]).ok_or_else(|| EvalError::TypeError("mse_loss expects a tensor".to_string()))?;
            let y_true = heap.get_tensor_mut(y_true_id)?.clone();
            let y_pred = heap.get_tensor_mut(y_pred_id)?.clone();
            let res = ops::mse_loss(y_true_id, &y_true, y_pred_id, &y_pred)?;
            let res_id = heap.register(HeapObject::Tensor(res));
            Ok(encode_heap_pointer(res_id))
        }
        "softmax_ce_loss" => {
            let y_true_id = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("softmax_ce_loss expects a tensor for y_true".to_string()))?;
            let logits_id = decode_heap_pointer(args[1]).ok_or_else(|| EvalError::TypeError("softmax_ce_loss expects a tensor for logits".to_string()))?;
            let y_true = heap.get_tensor_mut(y_true_id)?.clone();
            let logits = heap.get_tensor_mut(logits_id)?.clone();
            let res = ops::softmax_ce_loss(y_true_id, &y_true, logits_id, &logits)?;
            let res_id = heap.register(HeapObject::Tensor(res));
            Ok(encode_heap_pointer(res_id))
        }
        "flatten" => {
            let id = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("flatten expects a tensor".to_string()))?;
            let t = heap.get_tensor_mut(id)?.clone();
            let res = ops::flatten(id, &t)?;
            let res_id = heap.register(HeapObject::Tensor(res));
            Ok(encode_heap_pointer(res_id))
        }
        "sgd_update" => {
            let params_id = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("sgd_update expects a tensor".to_string()))?;
            let grads_id = decode_heap_pointer(args[1]).ok_or_else(|| EvalError::TypeError("sgd_update expects a tensor".to_string()))?;
            let learning_rate = args[2];
            let params = heap.get_tensor_mut(params_id)?.clone();
            let grads = heap.get_tensor_mut(grads_id)?.clone();
            let res = optimizers::sgd_update(&params, &grads, learning_rate)?;
            let res_id = heap.register(HeapObject::Tensor(res));
            Ok(encode_heap_pointer(res_id))
        }
        "adamw_init_state" => {
            let params_id = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("adamw_init_state expects a tensor".to_string()))?;
            let params = heap.get_tensor_mut(params_id)?.clone();
            let (m, v, t) = optimizers::adamw_init_state(&params);

            let m_id = heap.register(HeapObject::Tensor(m));
            let v_id = heap.register(HeapObject::Tensor(v));

            let t_cons = heap.register(HeapObject::Pair(t, NIL_VALUE));
            let v_cons = heap.register(HeapObject::Pair(encode_heap_pointer(v_id), encode_heap_pointer(t_cons)));
            let m_cons = heap.register(HeapObject::Pair(encode_heap_pointer(m_id), encode_heap_pointer(v_cons)));
            
            Ok(encode_heap_pointer(m_cons))
        }
        "adamw_update" => {
            let params_id = decode_heap_pointer(args[0]).ok_or_else(|| EvalError::TypeError("adamw_update: arg 1 (params) must be a tensor".to_string()))?;
            let grads_id = decode_heap_pointer(args[1]).ok_or_else(|| EvalError::TypeError("adamw_update: arg 2 (grads) must be a tensor".to_string()))?;
            
            let state_vec = list_to_vec(args[2], heap)?;
            if state_vec.len() != 3 { return Err(EvalError::TypeError("adamw_update: arg 3 (state) must be a list of 3 elements [m, v, t]".to_string())); }
            let m_id = decode_heap_pointer(state_vec[0]).ok_or_else(|| EvalError::TypeError("adamw_update: state element m must be a tensor".to_string()))?;
            let v_id = decode_heap_pointer(state_vec[1]).ok_or_else(|| EvalError::TypeError("adamw_update: state element v must be a tensor".to_string()))?;
            let t = state_vec[2];
            
            let lr = args[3];
            let beta1 = args[4];
            let beta2 = args[5];
            let epsilon = args[6];
            let weight_decay = args[7];

            let params = heap.get_tensor_mut(params_id)?.clone();
            let grads = heap.get_tensor_mut(grads_id)?.clone();
            let m = heap.get_tensor_mut(m_id)?.clone();
            let v = heap.get_tensor_mut(v_id)?.clone();

            let (new_params, new_m, new_v, new_t) = optimizers::adamw_update(&params, &grads, &m, &v, t, lr, beta1, beta2, epsilon, weight_decay)?;

            let new_params_id = heap.register(HeapObject::Tensor(new_params));
            let new_m_id = heap.register(HeapObject::Tensor(new_m));
            let new_v_id = heap.register(HeapObject::Tensor(new_v));

            let new_t_cons = heap.register(HeapObject::Pair(new_t, NIL_VALUE));
            let new_v_cons = heap.register(HeapObject::Pair(encode_heap_pointer(new_v_id), encode_heap_pointer(new_t_cons)));
            let new_m_cons = heap.register(HeapObject::Pair(encode_heap_pointer(new_m_id), encode_heap_pointer(new_v_cons)));

            let result_cons = heap.register(HeapObject::Pair(encode_heap_pointer(new_params_id), encode_heap_pointer(new_m_cons)));

            Ok(encode_heap_pointer(result_cons))
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