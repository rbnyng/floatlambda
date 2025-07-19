// src/vm/ml_natives.rs

use crate::interpreter::evaluator::{list_to_vec, vec_to_list};
use crate::memory::{decode_heap_pointer, encode_heap_pointer, HeapObject};
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
    NativeDef { name: "reshape", arity: 2, func: native_reshape },
    NativeDef { name: "transpose", arity: 1, func: native_transpose },
    NativeDef { name: "sum_t", arity: 1, func: native_sum_t },
    NativeDef { name: "mean_t", arity: 1, func: native_mean_t },
    NativeDef { name: "get_data", arity: 1, func: native_get_data },
    NativeDef { name: "get_shape", arity: 1, func: native_get_shape },
    NativeDef { name: "get_grad", arity: 1, func: native_get_grad },
    NativeDef { name: "grad", arity: 2, func: native_grad },
];

// src/vm/ml_natives.rs tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Heap, NIL_VALUE};
    use crate::vm::vm::VM;

    // Helper to set up a VM with a given heap for testing.
    // Leaking the heap gives it a static lifetime, which is fine for tests.
    fn setup_vm() -> VM<'static> {
        let heap = Box::leak(Box::new(Heap::new()));
        VM::new(heap)
    }
    
    // Helper to create a FloatLambda list on the heap from a Rust Vec.
    fn create_list_on_heap(vm: &mut VM, values: &[f64]) -> f64 {
        let mut list_ptr = NIL_VALUE;
        for &val in values.iter().rev() {
            let pair = HeapObject::Pair(val, list_ptr);
            let id = vm.heap.register(pair);
            list_ptr = encode_heap_pointer(id);
        }
        list_ptr
    }

    #[test]
    fn test_native_tensor_creation() {
        let mut vm = setup_vm();
        let shape_ptr = create_list_on_heap(&mut vm, &[2.0, 2.0]);
        let data_ptr = create_list_on_heap(&mut vm, &[1.0, 2.0, 3.0, 4.0]);

        vm.stack.push(shape_ptr);
        vm.stack.push(data_ptr);

        // Run the native function
        native_tensor(&mut vm).unwrap();

        // Check the result
        assert_eq!(vm.stack.len(), 1);
        let result_ptr = vm.stack.pop().unwrap();
        let result_id = decode_heap_pointer(result_ptr).expect("Result should be a heap pointer");

        match vm.heap.get(result_id) {
            Some(HeapObject::Tensor(t)) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
            }
            _ => panic!("Expected a Tensor on the heap"),
        }
    }

    #[test]
    fn test_native_add_t_op() {
        let mut vm = setup_vm();
        
        // Create two tensors on the heap
        let t1 = DifferentiableTensor::new(vec![2], vec![1.0, 2.0]);
        let t2 = DifferentiableTensor::new(vec![2], vec![10.0, 20.0]);
        let t1_id = vm.heap.register(HeapObject::Tensor(t1));
        let t2_id = vm.heap.register(HeapObject::Tensor(t2));

        // Push their pointers onto the stack
        vm.stack.push(encode_heap_pointer(t1_id));
        vm.stack.push(encode_heap_pointer(t2_id));
        
        // Run the native function
        native_add_t(&mut vm).unwrap();

        // Check the result
        let result_ptr = vm.stack.pop().unwrap();
        let result_id = decode_heap_pointer(result_ptr).unwrap();
        
        match vm.heap.get(result_id) {
            Some(HeapObject::Tensor(t)) => {
                assert_eq!(t.shape, vec![2]);
                assert_eq!(t.data, vec![11.0, 22.0]);
                // Check that the computation graph was created
                assert!(t.context.is_some());
                let ctx = t.context.as_ref().unwrap();
                assert_eq!(ctx.parents, vec![t1_id, t2_id]);
            }
            _ => panic!("Expected a Tensor on the heap"),
        }
    }

    #[test]
    fn test_native_get_data_op() {
        let mut vm = setup_vm();
        
        // Create a tensor on the heap
        let t1 = DifferentiableTensor::new(vec![2], vec![100.0, 200.0]);
        let t1_id = vm.heap.register(HeapObject::Tensor(t1));

        // Push its pointer onto the stack
        vm.stack.push(encode_heap_pointer(t1_id));

        // Run the native function
        native_get_data(&mut vm).unwrap();

        // The result should be a pointer to a new list on the heap
        let list_ptr = vm.stack.pop().unwrap();
        let result_vec = list_to_vec(list_ptr, vm.heap).unwrap();

        assert_eq!(result_vec, vec![100.0, 200.0]);
    }

    #[test]
    fn test_native_tensor_arg_order() {
        // This test specifically checks the stack pop order, which was a suspected bug.
        let mut vm = setup_vm();
        let shape_ptr = create_list_on_heap(&mut vm, &[1.0]); // Shape [1]
        let data_ptr = create_list_on_heap(&mut vm, &[42.0]); // Data [42.0]

        // For (tensor shape data), stack is [..., shape_ptr, data_ptr]
        // The native pops data_ptr then shape_ptr.
        vm.stack.push(shape_ptr);
        vm.stack.push(data_ptr);

        // Run the native function
        let result = native_tensor(&mut vm);
        assert!(result.is_ok(), "native_tensor failed with correct stack order");
        
        let tensor_ptr = vm.stack.pop().unwrap();
        let tensor_id = decode_heap_pointer(tensor_ptr).unwrap();
        match vm.heap.get(tensor_id) {
            Some(HeapObject::Tensor(t)) => {
                assert_eq!(t.shape, vec![1]);
                assert_eq!(t.data, vec![42.0]);
            }
            _ => panic!("Expected tensor"),
        }
    }
}