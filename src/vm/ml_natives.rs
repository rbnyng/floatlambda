// src/vm/ml_natives.rs

use crate::interpreter::evaluator::{list_to_vec, vec_to_list};
use crate::memory::{decode_heap_pointer, encode_heap_pointer, HeapObject, NIL_VALUE};
use crate::ml::{self, ops, DifferentiableTensor};
use crate::vm::natives::NativeDef;
use crate::vm::vm::{InterpretError, VM};

// --- Native Implementations ---

fn native_tensor(vm: &mut VM) -> Result<(), InterpretError> {
    let data_list_ptr = vm.pop_stack()?;
    let shape_list_ptr = vm.pop_stack()?;

    let shape_vec = list_to_vec(shape_list_ptr, vm.heap)
        .map_err(|e| InterpretError::Runtime(e.to_string()))?
        .iter()
        .map(|&x| x as usize)
        .collect();
    let data_vec = list_to_vec(data_list_ptr, vm.heap)
        .map_err(|e| InterpretError::Runtime(e.to_string()))?;

    let tensor = DifferentiableTensor::new(shape_vec, data_vec);
    let id = vm.heap.register(HeapObject::Tensor(tensor));
    vm.stack.push(encode_heap_pointer(id));
    Ok(())
}

fn native_add_t(vm: &mut VM) -> Result<(), InterpretError> {
    let t2_ptr = vm.pop_stack()?;
    let t1_ptr = vm.pop_stack()?;
    let id2 = decode_heap_pointer(t2_ptr).ok_or_else(|| InterpretError::Runtime("add_t expects a tensor".to_string()))?;
    let id1 = decode_heap_pointer(t1_ptr).ok_or_else(|| InterpretError::Runtime("add_t expects a tensor".to_string()))?;
    let t2 = vm.heap.get_tensor_mut(id2).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let t1 = vm.heap.get_tensor_mut(id1).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();

    let res = ops::add(id1, &t1, id2, &t2).map_err(|e| InterpretError::Runtime(e.to_string()))?;
    let res_id = vm.heap.register(HeapObject::Tensor(res));
    vm.stack.push(encode_heap_pointer(res_id));
    Ok(())
}

fn native_matmul(vm: &mut VM) -> Result<(), InterpretError> {
    let t2_ptr = vm.pop_stack()?;
    let t1_ptr = vm.pop_stack()?;
    let id2 = decode_heap_pointer(t2_ptr).ok_or_else(|| InterpretError::Runtime("matmul expects a tensor".to_string()))?;
    let id1 = decode_heap_pointer(t1_ptr).ok_or_else(|| InterpretError::Runtime("matmul expects a tensor".to_string()))?;
    let t2 = vm.heap.get_tensor_mut(id2).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let t1 = vm.heap.get_tensor_mut(id1).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();

    let res = ops::matmul(id1, &t1, id2, &t2).map_err(|e| InterpretError::Runtime(e.to_string()))?;
    let res_id = vm.heap.register(HeapObject::Tensor(res));
    vm.stack.push(encode_heap_pointer(res_id));
    Ok(())
}

fn native_sigmoid_t(vm: &mut VM) -> Result<(), InterpretError> {
    let t_ptr = vm.pop_stack()?;
    let id = decode_heap_pointer(t_ptr).ok_or_else(|| InterpretError::Runtime("sigmoid_t expects a tensor".to_string()))?;
    let t = vm.heap.get_tensor_mut(id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let res = ops::sigmoid(id, &t);
    let res_id = vm.heap.register(HeapObject::Tensor(res));
    vm.stack.push(encode_heap_pointer(res_id));
    Ok(())
}

fn native_relu_t(vm: &mut VM) -> Result<(), InterpretError> {
    let t_ptr = vm.pop_stack()?;
    let id = decode_heap_pointer(t_ptr).ok_or_else(|| InterpretError::Runtime("relu_t expects a tensor".to_string()))?;
    let t = vm.heap.get_tensor_mut(id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let res = ops::relu(id, &t);
    let res_id = vm.heap.register(HeapObject::Tensor(res));
    vm.stack.push(encode_heap_pointer(res_id));
    Ok(())
}

fn native_mse_loss(vm: &mut VM) -> Result<(), InterpretError> {
    let y_pred_ptr = vm.pop_stack()?;
    let y_true_ptr = vm.pop_stack()?;
    let y_pred_id = decode_heap_pointer(y_pred_ptr).ok_or_else(|| InterpretError::Runtime("mse_loss expects a tensor".to_string()))?;
    let y_true_id = decode_heap_pointer(y_true_ptr).ok_or_else(|| InterpretError::Runtime("mse_loss expects a tensor".to_string()))?;
    let y_pred = vm.heap.get_tensor_mut(y_pred_id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let y_true = vm.heap.get_tensor_mut(y_true_id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();

    let res = ops::mse_loss(y_true_id, &y_true, y_pred_id, &y_pred).map_err(|e| InterpretError::Runtime(e.to_string()))?;
    let res_id = vm.heap.register(HeapObject::Tensor(res));
    vm.stack.push(encode_heap_pointer(res_id));
    Ok(())
}

fn native_sgd_update(vm: &mut VM) -> Result<(), InterpretError> {
    let learning_rate = vm.pop_stack()?;
    let grads_ptr = vm.pop_stack()?;
    let params_ptr = vm.pop_stack()?;

    let grads_id = decode_heap_pointer(grads_ptr).ok_or_else(|| InterpretError::Runtime("sgd_update expects a tensor".to_string()))?;
    let params_id = decode_heap_pointer(params_ptr).ok_or_else(|| InterpretError::Runtime("sgd_update expects a tensor".to_string()))?;

    let grads = vm.heap.get_tensor_mut(grads_id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let params = vm.heap.get_tensor_mut(params_id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    
    let res = ml::optimizers::sgd_update(&params, &grads, learning_rate).map_err(|e| InterpretError::Runtime(e.to_string()))?;
    let res_id = vm.heap.register(HeapObject::Tensor(res));
    vm.stack.push(encode_heap_pointer(res_id));
    Ok(())
}

fn native_adamw_init_state(vm: &mut VM) -> Result<(), InterpretError> {
    let params_ptr = vm.pop_stack()?;
    let params_id = decode_heap_pointer(params_ptr).ok_or_else(|| InterpretError::Runtime("adamw_init_state expects a tensor".to_string()))?;
    let params = vm.heap.get_tensor_mut(params_id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    
    let (m, v, t) = ml::optimizers::adamw_init_state(&params);
    let m_id = vm.heap.register(HeapObject::Tensor(m));
    let v_id = vm.heap.register(HeapObject::Tensor(v));

    let t_cons = vm.heap.register(HeapObject::Pair(t, NIL_VALUE));
    let v_cons = vm.heap.register(HeapObject::Pair(encode_heap_pointer(v_id), encode_heap_pointer(t_cons)));
    let m_cons = vm.heap.register(HeapObject::Pair(encode_heap_pointer(m_id), encode_heap_pointer(v_cons)));
    
    vm.stack.push(encode_heap_pointer(m_cons));
    Ok(())
}

fn native_adamw_update(vm: &mut VM) -> Result<(), InterpretError> {
    let weight_decay = vm.pop_stack()?;
    let epsilon = vm.pop_stack()?;
    let beta2 = vm.pop_stack()?;
    let beta1 = vm.pop_stack()?;
    let lr = vm.pop_stack()?;
    let state_ptr = vm.pop_stack()?;
    let grads_ptr = vm.pop_stack()?;
    let params_ptr = vm.pop_stack()?;

    let params_id = decode_heap_pointer(params_ptr).ok_or_else(|| InterpretError::Runtime("adamw_update: arg 1 (params) must be a tensor".to_string()))?;
    let grads_id = decode_heap_pointer(grads_ptr).ok_or_else(|| InterpretError::Runtime("adamw_update: arg 2 (grads) must be a tensor".to_string()))?;
    
    let state_vec = list_to_vec(state_ptr, vm.heap).map_err(|e| InterpretError::Runtime(e.to_string()))?;
    if state_vec.len() != 3 { return Err(InterpretError::Runtime("adamw_update: arg 3 (state) must be a list of 3 elements [m, v, t]".to_string())); }
    let m_id = decode_heap_pointer(state_vec[0]).ok_or_else(|| InterpretError::Runtime("adamw_update: state element m must be a tensor".to_string()))?;
    let v_id = decode_heap_pointer(state_vec[1]).ok_or_else(|| InterpretError::Runtime("adamw_update: state element v must be a tensor".to_string()))?;
    let t = state_vec[2];
    
    let params = vm.heap.get_tensor_mut(params_id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let grads = vm.heap.get_tensor_mut(grads_id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let m = vm.heap.get_tensor_mut(m_id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let v = vm.heap.get_tensor_mut(v_id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();

    let (new_params, new_m, new_v, new_t) = ml::optimizers::adamw_update(&params, &grads, &m, &v, t, lr, beta1, beta2, epsilon, weight_decay)
        .map_err(|e| InterpretError::Runtime(e.to_string()))?;

    let new_params_id = vm.heap.register(HeapObject::Tensor(new_params));
    let new_m_id = vm.heap.register(HeapObject::Tensor(new_m));
    let new_v_id = vm.heap.register(HeapObject::Tensor(new_v));

    let new_t_cons = vm.heap.register(HeapObject::Pair(new_t, NIL_VALUE));
    let new_v_cons = vm.heap.register(HeapObject::Pair(encode_heap_pointer(new_v_id), encode_heap_pointer(new_t_cons)));
    let new_m_cons = vm.heap.register(HeapObject::Pair(encode_heap_pointer(new_m_id), encode_heap_pointer(new_v_cons)));
    let result_cons = vm.heap.register(HeapObject::Pair(encode_heap_pointer(new_params_id), encode_heap_pointer(new_m_cons)));

    vm.stack.push(encode_heap_pointer(result_cons));
    Ok(())
}

fn native_softmax_ce_loss(vm: &mut VM) -> Result<(), InterpretError> {
    let logits_ptr = vm.pop_stack()?;
    let y_true_ptr = vm.pop_stack()?;
    let logits_id = decode_heap_pointer(logits_ptr).ok_or_else(|| InterpretError::Runtime("softmax_ce_loss expects a tensor for logits".to_string()))?;
    let y_true_id = decode_heap_pointer(y_true_ptr).ok_or_else(|| InterpretError::Runtime("softmax_ce_loss expects a tensor for y_true".to_string()))?;
    let logits = vm.heap.get_tensor_mut(logits_id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let y_true = vm.heap.get_tensor_mut(y_true_id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();

    let res = ops::softmax_ce_loss(y_true_id, &y_true, logits_id, &logits).map_err(|e| InterpretError::Runtime(e.to_string()))?;
    let res_id = vm.heap.register(HeapObject::Tensor(res));
    vm.stack.push(encode_heap_pointer(res_id));
    Ok(())
}

fn native_flatten(vm: &mut VM) -> Result<(), InterpretError> {
    let t_ptr = vm.pop_stack()?;
    let id = decode_heap_pointer(t_ptr).ok_or_else(|| InterpretError::Runtime("flatten expects a tensor".to_string()))?;
    let t = vm.heap.get_tensor_mut(id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let res = ops::flatten(id, &t).map_err(|e| InterpretError::Runtime(e.to_string()))?;
    let res_id = vm.heap.register(HeapObject::Tensor(res));
    vm.stack.push(encode_heap_pointer(res_id));
    Ok(())
}

fn native_reshape(vm: &mut VM) -> Result<(), InterpretError> {
    let new_shape_ptr = vm.pop_stack()?;
    let t_ptr = vm.pop_stack()?;
    let id = decode_heap_pointer(t_ptr).ok_or_else(|| InterpretError::Runtime("reshape expects a tensor".to_string()))?;
    let new_shape_vec = list_to_vec(new_shape_ptr, vm.heap)
        .map_err(|e| InterpretError::Runtime(e.to_string()))?
        .iter().map(|&x| x as usize).collect();
    let tensor = vm.heap.get_tensor_mut(id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let res = ops::reshape(id, &tensor, new_shape_vec).map_err(|e| InterpretError::Runtime(e.to_string()))?;
    let res_id = vm.heap.register(HeapObject::Tensor(res));
    vm.stack.push(encode_heap_pointer(res_id));
    Ok(())
}

fn native_transpose(vm: &mut VM) -> Result<(), InterpretError> {
    let t_ptr = vm.pop_stack()?;
    let id = decode_heap_pointer(t_ptr).ok_or_else(|| InterpretError::Runtime("transpose expects a tensor".to_string()))?;
    let tensor = vm.heap.get_tensor_mut(id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let (new_shape, new_data) = ops::transpose(&tensor.shape, &tensor.data).map_err(|e| InterpretError::Runtime(e.to_string()))?;
    let res = DifferentiableTensor::new(new_shape, new_data);
    let res_id = vm.heap.register(HeapObject::Tensor(res));
    vm.stack.push(encode_heap_pointer(res_id));
    Ok(())
}

fn native_sum_t(vm: &mut VM) -> Result<(), InterpretError> {
    let t_ptr = vm.pop_stack()?;
    let id = decode_heap_pointer(t_ptr).ok_or_else(|| InterpretError::Runtime("sum_t expects a tensor".to_string()))?;
    let tensor = vm.heap.get_tensor_mut(id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let res = ops::sum_t(id, &tensor);
    let res_id = vm.heap.register(HeapObject::Tensor(res));
    vm.stack.push(encode_heap_pointer(res_id));
    Ok(())
}

fn native_mean_t(vm: &mut VM) -> Result<(), InterpretError> {
    let t_ptr = vm.pop_stack()?;
    let id = decode_heap_pointer(t_ptr).ok_or_else(|| InterpretError::Runtime("mean_t expects a tensor".to_string()))?;
    let tensor = vm.heap.get_tensor_mut(id).map_err(|e| InterpretError::Runtime(e.to_string()))?.clone();
    let res = ops::mean_t(id, &tensor);
    let res_id = vm.heap.register(HeapObject::Tensor(res));
    vm.stack.push(encode_heap_pointer(res_id));
    Ok(())
}

fn native_get_data(vm: &mut VM) -> Result<(), InterpretError> {
    let t_ptr = vm.pop_stack()?;
    let id = decode_heap_pointer(t_ptr).ok_or_else(|| InterpretError::Runtime("get_data expects a tensor".to_string()))?;
    let tensor = vm.heap.get_tensor_mut(id).map_err(|e| InterpretError::Runtime(e.to_string()))?;
    let data = tensor.data.clone();
    vm.stack.push(vec_to_list(&data, vm.heap));
    Ok(())
}

fn native_get_shape(vm: &mut VM) -> Result<(), InterpretError> {
    let t_ptr = vm.pop_stack()?;
    let id = decode_heap_pointer(t_ptr).ok_or_else(|| InterpretError::Runtime("get_shape expects a tensor".to_string()))?;
    let tensor = vm.heap.get_tensor_mut(id).map_err(|e| InterpretError::Runtime(e.to_string()))?;
    let shape_f64: Vec<f64> = tensor.shape.iter().map(|&x| x as f64).collect();
    vm.stack.push(vec_to_list(&shape_f64, vm.heap));
    Ok(())
}

fn native_get_grad(vm: &mut VM) -> Result<(), InterpretError> {
    let t_ptr = vm.pop_stack()?;
    let id = decode_heap_pointer(t_ptr).ok_or_else(|| InterpretError::Runtime("get_grad expects a tensor".to_string()))?;
    let tensor = vm.heap.get_tensor_mut(id).map_err(|e| InterpretError::Runtime(e.to_string()))?;
    let grad = tensor.grad.borrow().clone();
    vm.stack.push(vec_to_list(&grad, vm.heap));
    Ok(())
}

fn native_grad(vm: &mut VM) -> Result<(), InterpretError> {
    let input_tensor_ptr = vm.pop_stack()?;
    let func_ptr = vm.pop_stack()?;
    let input_tensor_id = decode_heap_pointer(input_tensor_ptr)
        .ok_or_else(|| InterpretError::Runtime("grad expects a tensor as the second argument".to_string()))?;
    // --- 1. Forward Pass using re-entrant VM call ---
    vm.stack.push(func_ptr);
    vm.stack.push(input_tensor_ptr);
    vm.call_value(func_ptr, 1)?;
    let output_tensor_ptr = vm.run()?;
    vm.pop_stack()?; // Clean up stack from run()

    let output_tensor_id = decode_heap_pointer(output_tensor_ptr)
        .ok_or_else(|| InterpretError::Runtime("grad function must return a tensor.".to_string()))?;
    // --- 2. Backward Pass (pure Rust logic) ---
    {
        let output_tensor = vm.heap.get_tensor_mut(output_tensor_id).map_err(|e| InterpretError::Runtime(e.to_string()))?;
        if output_tensor.data.len() != 1 {
            return Err(InterpretError::Runtime(format!(
                "grad function must return a scalar tensor (length 1), but got shape {:?}",
                output_tensor.shape
            )));
        }
        output_tensor.grad.borrow_mut()[0] = 1.0;
    }
    
    let topo_order = ml::autodiff::build_topo_order(output_tensor_id, vm.heap);
    for &node_id in topo_order.iter().rev() {
        let (context, grad_data) = {
            let tensor = vm.heap.get_tensor_mut(node_id).map_err(|e| InterpretError::Runtime(e.to_string()))?;
            (tensor.context.clone(), tensor.grad.borrow().clone())
        };
        if let Some(ctx) = context {
            (ctx.backward_fn)(&grad_data, vm.heap);
        }
    }
    
    // --- 3. Return the result ---
    let (final_grad_shape, final_grad_data) = {
        let input_tensor = vm.heap.get_tensor_mut(input_tensor_id).map_err(|e| InterpretError::Runtime(e.to_string()))?;
        (input_tensor.shape.clone(), input_tensor.grad.borrow().clone())
    };
    
    let grad_tensor = DifferentiableTensor::new(final_grad_shape, final_grad_data);
    let grad_id = vm.heap.register(HeapObject::Tensor(grad_tensor));
    vm.stack.push(encode_heap_pointer(grad_id));
    Ok(())
}

// --- The Native Definition Table ---

pub const ML_NATIVES: &[NativeDef] = &[
    NativeDef { name: "tensor", arity: 2, func: native_tensor },
    NativeDef { name: "add_t", arity: 2, func: native_add_t },
    NativeDef { name: "matmul", arity: 2, func: native_matmul },
    NativeDef { name: "sigmoid_t", arity: 1, func: native_sigmoid_t },
    NativeDef { name: "relu_t", arity: 1, func: native_relu_t },
    NativeDef { name: "mse_loss", arity: 2, func: native_mse_loss },
    NativeDef { name: "sgd_update", arity: 3, func: native_sgd_update },
    NativeDef { name: "adamw_init_state", arity: 1, func: native_adamw_init_state },
    NativeDef { name: "adamw_update", arity: 8, func: native_adamw_update },
    NativeDef { name: "softmax_ce_loss", arity: 2, func: native_softmax_ce_loss },
    NativeDef { name: "flatten", arity: 1, func: native_flatten },
    NativeDef { name: "reshape", arity: 2, func: native_reshape },
    NativeDef { name: "transpose", arity: 1, func: native_transpose },
    NativeDef { name: "sum_t", arity: 1, func: native_sum_t },
    NativeDef { name: "mean_t", arity: 1, func: native_mean_t },
    NativeDef { name: "get_data", arity: 1, func: native_get_data },
    NativeDef { name: "get_shape", arity: 1, func: native_get_shape },
    NativeDef { name: "get_grad", arity: 1, func: native_get_grad },
    NativeDef { name: "grad", arity: 2, func: native_grad },
];