// src/ml/autodiff.rs

use std::collections::HashSet;

use super::tensor::DifferentiableTensor;
use crate::interpreter::evaluator::apply_function;
use crate::memory::{decode_heap_pointer, encode_heap_pointer, Heap, HeapObject};
use crate::error::EvalError;

// Performs a topological sort of the computation graph starting from a root node.
pub fn build_topo_order(root_id: u64, heap: &Heap) -> Vec<u64> {
    let mut order = Vec::new();
    let mut visited = HashSet::new();

    fn visit(id: u64, heap: &Heap, visited: &mut HashSet<u64>, order: &mut Vec<u64>) {
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

        if let Some(HeapObject::Tensor(tensor)) = heap.get(id) {
            if let Some(ctx) = &tensor.context {
                for &parent_id in &ctx.parents {
                    visit(parent_id, heap, visited, order);
                }
            }
        }
        order.push(id);
    }
    
    visit(root_id, heap, &mut visited, &mut order);
    order
}

// The implementation of the grad builtin.
pub fn grad(func_ptr: f64, input_tensor_id: u64, heap: &mut Heap) -> Result<DifferentiableTensor, EvalError> {
    // 1. Run the forward pass to build the computation graph.
    let output_tensor_ptr = apply_function(func_ptr, encode_heap_pointer(input_tensor_id), heap)?;
    let output_tensor_id = decode_heap_pointer(output_tensor_ptr)
        .ok_or_else(|| EvalError::TypeError("grad function must return a tensor.".to_string()))?;

    // Check that the output is a scalar tensor for loss calculation
    let output_tensor = heap.get_tensor_mut(output_tensor_id)?.clone();
    if output_tensor.data.len() != 1 {
        return Err(EvalError::TypeError(format!(
            "grad function must return a scalar tensor (length 1), but got shape {:?}",
            output_tensor.shape
        )));
    }
    
    // 2. Build the topological order for backpropagation.
    let topo_order = build_topo_order(output_tensor_id, heap);

    // 3. Initialize the gradient of the final output to 1.0.
    heap.get_tensor_mut(output_tensor_id).unwrap().grad.borrow_mut()[0] = 1.0;
    
    // 4. Go backward through the graph and apply the chain rule.
    for &node_id in topo_order.iter().rev() {
        let (context, grad_data) = {
            let tensor = heap.get_tensor_mut(node_id)?;
            (tensor.context.clone(), tensor.grad.borrow().clone())
        };

        if let Some(ctx) = context {
            (ctx.backward_fn)(&grad_data, heap);
        }
    }

    // 5. The gradient is now stored in the original input tensor's .grad field.
    // Return this as a new, separate tensor.
    let final_grad_data = heap.get_tensor_mut(input_tensor_id).unwrap().grad.borrow().clone();
    let final_grad_shape = heap.get_tensor_mut(input_tensor_id).unwrap().shape.clone();
    
    Ok(DifferentiableTensor::new(final_grad_shape, final_grad_data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Term;
    use crate::ml::ops;
    use std::collections::HashMap;
    use std::rc::Rc;

    // Helper to set up a heap for a test
    fn setup_heap_with_tensor(tensor: DifferentiableTensor) -> (Heap, u64) {
        let mut heap = Heap::new();
        let id = heap.register(HeapObject::Tensor(tensor));
        (heap, id)
    }

    #[test]
    fn test_grad_of_add() {
        // Let f(T) = add(T, T) = 2*T. Loss is sum(2*T). Grad should be 2 everywhere.
        let t_start = DifferentiableTensor::new(vec![2], vec![10.0, 20.0]);
        let (mut heap, t_id) = setup_heap_with_tensor(t_start);

        // Mock the function f(T)
        let func = Term::Lam(
            "t".to_string(),
            Box::new(Term::App(
                Box::new(Term::App(
                    Box::new(Term::Builtin("add_t".to_string())),
                    Box::new(Term::Var("t".to_string())),
                )),
                Box::new(Term::Var("t".to_string())),
            )),
        );
        let env = Rc::new(HashMap::new());
        let _func_ptr = func.eval(&env, &mut heap).unwrap();
        
        // This is a stand-in for a real loss function.
        // We need to make it return a scalar. Let's imagine a sum_t builtin.
        // For this test, we'll manually implement the backprop logic.
        
        // --- Manual Forward and Backward Pass ---
        // Forward pass:
        let t1 = heap.get_tensor_mut(t_id).unwrap().clone();
        let t2 = heap.get_tensor_mut(t_id).unwrap().clone();
        let result_tensor = ops::add(t_id, &t1, t_id, &t2).unwrap();
        let result_id = heap.register(HeapObject::Tensor(result_tensor));

        // Backward pass:
        let topo_order = build_topo_order(result_id, &heap);
        heap.get_tensor_mut(result_id).unwrap().grad.borrow_mut()[0] = 1.0;
        heap.get_tensor_mut(result_id).unwrap().grad.borrow_mut()[1] = 1.0;
        
        for &node_id in topo_order.iter().rev() {
            let (context, grad_data) = {
                let tensor = heap.get_tensor_mut(node_id).unwrap();
                (tensor.context.clone(), tensor.grad.borrow().clone())
            };
            if let Some(ctx) = context {
                (ctx.backward_fn)(&grad_data, &mut heap);
            }
        }
        
        let final_grad = heap.get_tensor_mut(t_id).unwrap().grad.borrow().clone();
        assert_eq!(final_grad, vec![2.0, 2.0]);
    }
    
    #[test]
    fn test_grad_of_matmul() {
        // f(W) = matmul(X, W). Loss is sum(Y_hat).
        // Let X = [[1, 2]], W = [[3], [4]]. Y_hat = [[1*3 + 2*4]] = [[11]].
        // d(Y_hat)/dW = X^T = [[1], [2]]
        let x_tensor = DifferentiableTensor::new(vec![1, 2], vec![1.0, 2.0]);
        let w_tensor = DifferentiableTensor::new(vec![2, 1], vec![3.0, 4.0]);
        let (mut heap, x_id) = setup_heap_with_tensor(x_tensor);
        let w_id = heap.register(HeapObject::Tensor(w_tensor));

        // --- Manual Forward and Backward Pass ---
        let x = heap.get_tensor_mut(x_id).unwrap().clone();
        let w = heap.get_tensor_mut(w_id).unwrap().clone();
        let result_tensor = ops::matmul(x_id, &x, w_id, &w).unwrap();
        let result_id = heap.register(HeapObject::Tensor(result_tensor));

        let topo_order = build_topo_order(result_id, &heap);
        heap.get_tensor_mut(result_id).unwrap().grad.borrow_mut()[0] = 1.0;

        for &node_id in topo_order.iter().rev() {
            let (context, grad_data) = {
                let tensor = heap.get_tensor_mut(node_id).unwrap();
                (tensor.context.clone(), tensor.grad.borrow().clone())
            };
            if let Some(ctx) = context {
                (ctx.backward_fn)(&grad_data, &mut heap);
            }
        }

        let w_grad = heap.get_tensor_mut(w_id).unwrap().grad.borrow().clone();
        assert_eq!(w_grad, vec![1.0, 2.0]);

        // The gradient for X should be W^T = [[3, 4]]
        let x_grad = heap.get_tensor_mut(x_id).unwrap().grad.borrow().clone();
        assert_eq!(x_grad, vec![3.0, 4.0]);
    }

}