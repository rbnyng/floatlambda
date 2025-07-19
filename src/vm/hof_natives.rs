// src/vm/hof_natives.rs

use crate::memory::{decode_heap_pointer, encode_heap_pointer, HeapObject, NIL_VALUE};
use crate::vm::natives::NativeDef;
use crate::vm::vm::{InterpretError, VM};

// --- Native Implementations for Higher-Order Functions ---

fn native_length(vm: &mut VM) -> Result<(), InterpretError> {
    let mut current_list = vm.pop_stack()?;
    let mut count = 0.0;

    loop {
        if current_list == NIL_VALUE {
            break;
        }

        if let Some(id) = decode_heap_pointer(current_list) {
            if let Some(HeapObject::Pair(_, cdr)) = vm.heap.get(id) {
                count += 1.0;
                current_list = *cdr;
            } else {
                return Err(InterpretError::Runtime("'length' expects a proper list.".to_string()));
            }
        } else {
            return Err(InterpretError::Runtime("'length' expects a proper list.".to_string()));
        }
    }
    vm.stack.push(count);
    Ok(())
}

fn native_map(vm: &mut VM) -> Result<(), InterpretError> {
    let mut current_list = vm.pop_stack()?;
    let func_ptr = vm.pop_stack()?;

    let mut results = Vec::new();

    loop {
        if current_list == NIL_VALUE {
            break;
        }

        let (car, cdr) = if let Some(id) = decode_heap_pointer(current_list) {
            if let Some(HeapObject::Pair(car, cdr)) = vm.heap.get(id) {
                (*car, *cdr)
            } else {
                return Err(InterpretError::Runtime("'map' expects a proper list.".to_string()));
            }
        } else {
            return Err(InterpretError::Runtime("'map' expects a proper list.".to_string()));
        };

        // Re-entrant call to the VM to apply the function
        vm.stack.push(func_ptr);
        vm.stack.push(car);
        vm.call_value(func_ptr, 1)?;
        let mapped_val = vm.run()?;
        vm.pop_stack()?; // Clean up stack

        results.push(mapped_val);
        current_list = cdr;
    }

    // Build the new list from results
    let mut new_list = NIL_VALUE;
    for val in results.iter().rev() {
        let new_pair = HeapObject::Pair(*val, new_list);
        let new_id = vm.heap.register(new_pair);
        new_list = encode_heap_pointer(new_id);
    }
    vm.stack.push(new_list);
    Ok(())
}

fn native_filter(vm: &mut VM) -> Result<(), InterpretError> {
    let mut current_list = vm.pop_stack()?;
    let predicate_ptr = vm.pop_stack()?;

    let mut results = Vec::new();

    loop {
        if current_list == NIL_VALUE {
            break;
        }

        let (car, cdr) = if let Some(id) = decode_heap_pointer(current_list) {
            if let Some(HeapObject::Pair(car, cdr)) = vm.heap.get(id) {
                (*car, *cdr)
            } else {
                return Err(InterpretError::Runtime("'filter' expects a proper list.".to_string()));
            }
        } else {
            return Err(InterpretError::Runtime("'filter' expects a proper list.".to_string()));
        };

        // Re-entrant call to the VM
        vm.stack.push(predicate_ptr);
        vm.stack.push(car);
        vm.call_value(predicate_ptr, 1)?;
        let should_keep = vm.run()?;
        vm.pop_stack()?; // Clean up stack

        if should_keep != 0.0 && should_keep != NIL_VALUE {
            results.push(car);
        }
        current_list = cdr;
    }

    // Build the new list
    let mut new_list = NIL_VALUE;
    for val in results.iter().rev() {
        let new_pair = HeapObject::Pair(*val, new_list);
        let new_id = vm.heap.register(new_pair);
        new_list = encode_heap_pointer(new_id);
    }
    vm.stack.push(new_list);
    Ok(())
}

fn native_foldl(vm: &mut VM) -> Result<(), InterpretError> {
    let mut current_list = vm.pop_stack()?;
    let mut acc = vm.pop_stack()?;
    let func_ptr = vm.pop_stack()?;

    loop {
        if current_list == NIL_VALUE {
            break;
        }

        let (car, cdr) = if let Some(id) = decode_heap_pointer(current_list) {
            if let Some(HeapObject::Pair(car, cdr)) = vm.heap.get(id) {
                (*car, *cdr)
            } else {
                return Err(InterpretError::Runtime("'foldl' expects a proper list.".to_string()));
            }
        } else {
            return Err(InterpretError::Runtime("'foldl' expects a proper list.".to_string()));
        };
        
        // This is a 2-argument function call, so we need two re-entrant calls
        // 1. Apply the function to the accumulator
        vm.stack.push(func_ptr);
        vm.stack.push(acc);
        vm.call_value(func_ptr, 1)?;
        let partial_app_ptr = vm.run()?;
        vm.pop_stack()?;

        // 2. Apply the partial application to the current element
        vm.stack.push(partial_app_ptr);
        vm.stack.push(car);
        vm.call_value(partial_app_ptr, 1)?;
        acc = vm.run()?;
        vm.pop_stack()?;

        current_list = cdr;
    }

    vm.stack.push(acc);
    Ok(())
}

pub const HOF_NATIVES: &[NativeDef] = &[
    NativeDef { name: "length", arity: 1, func: native_length },
    NativeDef { name: "map", arity: 2, func: native_map },
    NativeDef { name: "filter", arity: 2, func: native_filter },
    NativeDef { name: "foldl", arity: 3, func: native_foldl },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Heap, NIL_VALUE};
    use crate::vm::{compiler, vm::VM};
    use crate::parser;

    // Helper to set up a VM with a given heap for testing.
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

    // Helper to compile a function and load it onto the heap.
    // Returns a pointer to the compiled closure.
    fn compile_and_load_fn(vm: &mut VM, source: &str) -> f64 {
        // Save stack size to restore it later, ensuring the helper doesn't alter test state.
        let initial_stack_size = vm.stack.len();

        // 1. Compile the source into a top-level script closure
        let script_closure_id = vm.compile_and_load(source)
            .expect("Test function compilation failed");

        // 2. Run the script. This executes the top-level code, which should evaluate
        //    to a function pointer, and returns that pointer.
        let result_ptr = vm.prime_and_run(script_closure_id)
            .expect("Test function script execution failed");

        // 3. Clean up the stack. prime_and_run leaves the result on the stack.
        vm.stack.truncate(initial_stack_size);
        
        // 4. Return the pointer to the actual function we want to test.
        result_ptr
    }

    #[test]
    fn test_native_length() {
        let mut vm = setup_vm();
        
        // Test with a list of 3 elements
        let list_ptr = create_list_on_heap(&mut vm, &[10.0, 20.0, 30.0]);
        vm.stack.push(list_ptr);
        native_length(&mut vm).unwrap();
        assert_eq!(vm.stack.pop().unwrap(), 3.0);

        // Test with an empty list
        let nil_ptr = NIL_VALUE;
        vm.stack.push(nil_ptr);
        native_length(&mut vm).unwrap();
        assert_eq!(vm.stack.pop().unwrap(), 0.0);
    }
    
    #[test]
    fn test_native_map() {
        let mut vm = setup_vm();
        
        // 1. Compile the mapping function: λx.(+ x 10)
        let add10_fn_ptr = compile_and_load_fn(&mut vm, "(λx.(+ x 10))");

        // 2. Create the input list: (cons 1 (cons 2 nil))
        let input_list_ptr = create_list_on_heap(&mut vm, &[1.0, 2.0]);
        
        // 3. Set up the stack for `(map add10_fn input_list)`
        vm.stack.push(add10_fn_ptr);
        vm.stack.push(input_list_ptr);
        
        // 4. Run the native map function
        native_map(&mut vm).unwrap();
        
        // 5. Verify the result
        let result_list_ptr = vm.stack.pop().unwrap();
        let result_vec = crate::interpreter::evaluator::list_to_vec(result_list_ptr, vm.heap).unwrap();
        
        assert_eq!(result_vec, vec![11.0, 12.0]);
    }
    
    #[test]
    fn test_native_foldl() {
        let mut vm = setup_vm();

        // 1. Compile the folding function: λacc.λx.(+ acc x)
        let add_fn_ptr = compile_and_load_fn(&mut vm, "(λacc.(λx.(+ acc x)))");

        // 2. Create the input list: (cons 10 (cons 20 (cons 30 nil)))
        let input_list_ptr = create_list_on_heap(&mut vm, &[10.0, 20.0, 30.0]);
        
        // 3. Set up the stack for `(foldl add_fn 0 initial_list)`
        vm.stack.push(add_fn_ptr);
        vm.stack.push(0.0); // Initial accumulator
        vm.stack.push(input_list_ptr);
        
        // 4. Run the native foldl function
        native_foldl(&mut vm).unwrap();

        // 5. Verify the result
        let result = vm.stack.pop().unwrap();
        assert_eq!(result, 60.0); // 0 + 10 + 20 + 30
    }
}