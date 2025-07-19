// tests/gc_tests.rs

use float_lambda::memory::{encode_heap_pointer, Heap, HeapObject, NIL_VALUE};
use float_lambda::vm::closure::Closure as VMClosure;
use float_lambda::vm::function::Function;
use float_lambda::vm::vm::{CallFrame, VM};
use float_lambda::{self as fl};

/// Helper to create a simple UserFunc (AST-based closure) for testing.
// This function needs a mutable VM reference to register the object on its heap.
fn create_dummy_ast_closure(vm: &mut VM) -> u64 {
    let closure = fl::memory::ASTClosure {
        param: "x".to_string(),
        body: Box::new(fl::ast::Term::Var("x".to_string())),
        env: std::rc::Rc::new(std::collections::HashMap::new()),
    };
    vm.heap.register(HeapObject::UserFunc(closure))
}

/// Helper to create a simple VM Function and Closure for testing.
// This function also needs a mutable VM reference.
fn create_dummy_vm_closure(vm: &mut VM) -> u64 {
    let func = Function {
        name: "dummy".to_string(),
        ..Default::default()
    };
    let func_id = vm.heap.register(HeapObject::Function(func));
    let closure = VMClosure {
        func_id,
        upvalues: std::rc::Rc::new(Vec::new()),
    };
    vm.heap.register(HeapObject::Closure(closure))
}

// ===================================================================
// 1. GC Correctness Tests (Don't Collect Live Objects)
// ===================================================================
#[cfg(test)]
mod gc_correctness_tests {
    use super::*;

    #[test]
    fn test_simple_root_preservation() {
        let mut heap = Heap::new();
        let id = heap.register(HeapObject::Pair(1.0, NIL_VALUE));
        assert_eq!(heap.alive_count(), 1);

        heap.collect_full(&[encode_heap_pointer(id)]);
        assert_eq!(heap.alive_count(), 1);
        assert!(matches!(heap.get(id), Some(HeapObject::Pair(_, _))));
    }

    #[test]
    fn test_transitive_reachability() {
        let mut heap = Heap::new();
        let id_c = heap.register(HeapObject::Pair(3.0, NIL_VALUE));
        let id_b = heap.register(HeapObject::Pair(2.0, encode_heap_pointer(id_c)));
        let id_a = heap.register(HeapObject::Pair(1.0, encode_heap_pointer(id_b)));
        assert_eq!(heap.alive_count(), 3);

        // Only root 'A'. 'B' and 'C' should be found by tracing.
        heap.collect_full(&[encode_heap_pointer(id_a)]);

        assert_eq!(heap.alive_count(), 3);
        assert!(heap.get(id_a).is_some());
        assert!(heap.get(id_b).is_some());
        assert!(heap.get(id_c).is_some());
    }

    #[test]
    fn test_closure_environment_preservation() {
        let mut heap = Heap::new();
        let vm = VM::new(&mut heap); // Create VM to access its heap

        // Create an environment that contains a pointer to a pair.
        let pair_id = vm.heap.register(HeapObject::Pair(42.0, NIL_VALUE));
        let mut env_map = std::collections::HashMap::new();
        env_map.insert("captured_val".to_string(), encode_heap_pointer(pair_id));

        // Create a closure with this environment.
        let closure = fl::memory::ASTClosure {
            param: "x".to_string(),
            body: Box::new(fl::ast::Term::Var("x".to_string())),
            env: std::rc::Rc::new(env_map),
        };
        let closure_id = vm.heap.register(HeapObject::UserFunc(closure));

        assert_eq!(vm.heap.alive_count(), 2); // The closure and the pair.

        // Collect garbage with only the closure as a root.
        vm.heap.collect_full(&[encode_heap_pointer(closure_id)]);

        // Both the closure and the pair it references should survive.
        assert_eq!(vm.heap.alive_count(), 2);
        assert!(vm.heap.get(closure_id).is_some());
        assert!(vm.heap.get(pair_id).is_some());
    }
}

// ===================================================================
// 2. GC Completeness Tests (Collect All Garbage)
// ===================================================================
#[cfg(test)]
mod gc_completeness_tests {
    use super::*;

    #[test]
    fn test_simple_unrooted_object() {
        let mut heap = Heap::new();
        let _id = heap.register(HeapObject::Pair(1.0, NIL_VALUE));
        assert_eq!(heap.alive_count(), 1);

        // Collect with no roots.
        heap.collect_full(&[]);
        assert_eq!(heap.alive_count(), 0);
    }

    #[test]
    fn test_reclaiming_part_of_a_structure() {
        let mut heap = Heap::new();
        let id_c = heap.register(HeapObject::Pair(3.0, NIL_VALUE));
        let id_b = heap.register(HeapObject::Pair(2.0, encode_heap_pointer(id_c)));
        let id_a = heap.register(HeapObject::Pair(1.0, encode_heap_pointer(id_b)));
        assert_eq!(heap.alive_count(), 3);

        // Only root 'B'. 'A' should be collected.
        heap.collect_full(&[encode_heap_pointer(id_b)]);

        assert_eq!(heap.alive_count(), 2);
        assert!(matches!(heap.get(id_a), Some(HeapObject::Free(_)) | None));
        assert!(heap.get(id_b).is_some());
        assert!(heap.get(id_c).is_some());
    }

    #[test]
    fn test_circular_references() {
        let mut heap = Heap::new();

        // Create a cycle: A -> B -> A
        // We need to allocate them first, then mutate them.
        let id_a = heap.register(HeapObject::Pair(1.0, NIL_VALUE));
        let id_b = heap.register(HeapObject::Pair(2.0, NIL_VALUE));

        // Create the cycle
        if let Some(HeapObject::Pair(_, cdr_a)) = heap.get_mut(id_a) {
            *cdr_a = encode_heap_pointer(id_b);
        }
        if let Some(HeapObject::Pair(_, cdr_b)) = heap.get_mut(id_b) {
            *cdr_b = encode_heap_pointer(id_a);
        }

        assert_eq!(heap.alive_count(), 2);

        // Collect with no roots. A tracing GC should correctly handle the cycle.
        heap.collect_full(&[]);
        assert_eq!(heap.alive_count(), 0);
    }
}

// ===================================================================
// 3. VM Integration Tests (Rooting the VM's Live State)
// ===================================================================
#[cfg(test)]
mod gc_vm_integration_tests {
    use super::*;

    #[test]
    fn test_gc_roots_the_vm_stack() {
        let mut heap = Heap::new();
        let mut vm = VM::new(&mut heap); // vm borrows heap mutably

        let id1 = create_dummy_ast_closure(&mut vm); // Pass vm to helpers
        let id2 = vm.heap.register(HeapObject::Pair(10.0, 20.0)); // Access heap through vm.heap

        // Push some values, including pointers, onto the VM stack.
        vm.stack.push(1.0);
        vm.stack.push(encode_heap_pointer(id1));
        vm.stack.push(2.0);
        vm.stack.push(encode_heap_pointer(id2));

        assert_eq!(vm.heap.alive_count(), 2); // Access heap through vm.heap

        // Collect, rooting only the stack (and globals, which are empty).
        let mut roots: Vec<f64> = vm.globals.values().copied().collect();
        roots.extend_from_slice(&vm.stack);
        vm.heap.collect_full(&roots); // Access heap through vm.heap

        // Everything on the stack should have survived.
        assert_eq!(vm.heap.alive_count(), 2); // Access heap through vm.heap
        assert!(vm.heap.get(id1).is_some()); // Access heap through vm.heap
        assert!(vm.heap.get(id2).is_some()); // Access heap through vm.heap
    }

    #[test]
    fn test_gc_roots_the_vm_call_frames() {
        let mut heap = Heap::new();
        let mut vm = VM::new(&mut heap); // vm borrows heap mutably

        let closure_id1 = create_dummy_vm_closure(&mut vm); // Pass vm to helpers
        let closure_id2 = create_dummy_vm_closure(&mut vm); // Pass vm to helpers
        let unrelated_obj_id = vm.heap.register(HeapObject::Pair(0.0, 0.0)); // Access heap through vm.heap

        // Push two call frames onto the VM.
        vm.frames.push(CallFrame {
            closure_id: closure_id1,
            ip: 0,
            stack_slot: 0,
        });
        vm.frames.push(CallFrame {
            closure_id: closure_id2,
            ip: 0,
            stack_slot: 5,
        });

        // Heap contains 2 closures, 2 functions, 1 pair = 5 objects
        assert_eq!(vm.heap.alive_count(), 5); // Access heap through vm.heap

        // Collect, rooting only the call frames.
        let mut roots = Vec::new();
        for frame in &vm.frames {
            roots.push(encode_heap_pointer(frame.closure_id));
        }
        vm.heap.collect_full(&roots); // Access heap through vm.heap

        // The two closures and their associated functions should survive.
        // The unrelated pair should be collected.
        assert_eq!(vm.heap.alive_count(), 4); // Access heap through vm.heap
        assert!(vm.heap.get(closure_id1).is_some()); // Access heap through vm.heap
        assert!(vm.heap.get(closure_id2).is_some()); // Access heap through vm.heap
        assert!(matches!(
            vm.heap.get(unrelated_obj_id),
            Some(HeapObject::Free(_)) | None
        )); // Access heap through vm.heap
    }

    /// It simulates a long-running script (prelude) that can trigger the GC,
    /// followed by another script that uses the results of the first.
    /// This requires the REPL's GC rooting strategy to be correct.
    #[test]
    fn test_gc_does_not_corrupt_vm_state_during_execution() {
        let mut heap = Heap::new();
        let mut vm = VM::new(&mut heap);

        // 1. Load the prelude.
        let prelude_code = include_str!("../src/prelude.fl");
        let prelude_id = vm.compile_and_load(prelude_code).unwrap();
        let prelude_result = vm.prime_and_run(prelude_id);
        assert!(prelude_result.is_ok(), "Prelude failed to load: {:?}", prelude_result.err());
        
        // Clear stack and frames to simulate idle REPL state.
        vm.stack.clear(); 
        vm.frames.clear();

        let identity_ptr_before_gc = vm.globals.get("identity").cloned().unwrap();
        let identity_id_before_gc = float_lambda::memory::decode_heap_pointer(identity_ptr_before_gc).unwrap();

        // 2. Simulate the REPL loop's GC step.
        let roots: Vec<f64> = vm.globals.values().copied().collect();
        vm.heap.collect_full(&roots);

        // 3. Assert that our known global survived.
        let identity_ptr_after_gc = vm.globals.get("identity").cloned().unwrap();
        assert_eq!(identity_ptr_before_gc.to_bits(), identity_ptr_after_gc.to_bits(), "Pointer to 'identity' global changed after GC");
        assert!(vm.heap.get(identity_id_before_gc).is_some(), "'identity' closure was incorrectly collected");
        
        // 4. Now, run the code, which relies on the prelude being intact.
        let script = "let rec loop = (λn. if (< n 1) then nil else (cons n (loop (- n 1)))) in (length (loop 100))";
        let script_id = vm.compile_and_load(script).unwrap();
        let result = vm.prime_and_run(script_id);

        // 5. Assert that the script ran successfully.
        assert!(result.is_ok(), "Script failed with error: {:?}", result.err());
        assert_eq!(result.unwrap(), 100.0);
    }
}
