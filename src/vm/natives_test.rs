// src/vm/natives_test.rs

use super::natives::*;
use super::vm::{InterpretError, VM};
use crate::memory::{encode_heap_pointer, Heap, HeapObject, NIL_VALUE};

// Helper to set up a VM with a given stack for testing native calls
fn setup_vm_with_stack(stack_vals: &[f64]) -> VM<'static> {
    // Leaking the heap is a simple way to get a 'static lifetime for tests.
    // This is fine for unit testing but should not be done in production code.
    let heap = Box::leak(Box::new(Heap::new()));
    let mut vm = VM::new(heap);
    vm.stack.extend_from_slice(stack_vals);
    vm
}

// Helper to check for approximate float equality, handling NaN correctly.
fn assert_float_eq(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "Expected NaN, but got {}", actual);
    } else {
        assert!(
            (actual - expected).abs() < 1e-9,
            "Assertion failed: (actual: {} - expected: {}).abs() < 1e-9",
            actual,
            expected
        );
    }
}

// --- Nullary Constant Tests ---
#[test]
fn test_native_pi() {
    let mut vm = setup_vm_with_stack(&[]);
    let pi_native_def = &NATIVES[NATIVE_MAP["pi"].0 as usize];
    (pi_native_def.func)(&mut vm).unwrap();
    assert_eq!(pi_native_def.arity, 0);
    assert_float_eq(vm.pop_stack().unwrap(), std::f64::consts::PI);
}

#[test]
fn test_native_e() {
    let mut vm = setup_vm_with_stack(&[]);
    let e_native_def = &NATIVES[NATIVE_MAP["e"].0 as usize];
    (e_native_def.func)(&mut vm).unwrap();
    assert_eq!(e_native_def.arity, 0);
    assert_float_eq(vm.pop_stack().unwrap(), std::f64::consts::E);
}

// --- Nullary Pure Function Tests ---
#[test]
fn test_native_random() {
    let mut vm = setup_vm_with_stack(&[]);
    let random_native_def = &NATIVES[NATIVE_MAP["random"].0 as usize];
    (random_native_def.func)(&mut vm).unwrap();
    assert_eq!(random_native_def.arity, 0);
    let result = vm.pop_stack().unwrap();
    assert!(result >= 0.0 && result < 1.0);
}

// --- Unary Pure Function Tests ---
#[test]
fn test_native_sin() {
    let mut vm = setup_vm_with_stack(&[std::f64::consts::PI / 2.0]);
    let sin_native_def = &NATIVES[NATIVE_MAP["sin"].0 as usize];
    (sin_native_def.func)(&mut vm).unwrap();
    assert_eq!(sin_native_def.arity, 1);
    assert_float_eq(vm.pop_stack().unwrap(), 1.0);
}

#[test]
fn test_native_sqrt() {
    let sqrt_native_def = &NATIVES[NATIVE_MAP["sqrt"].0 as usize];
    assert_eq!(sqrt_native_def.arity, 1);

    // Positive case
    let mut vm_pos = setup_vm_with_stack(&[16.0]);
    (sqrt_native_def.func)(&mut vm_pos).unwrap();
    assert_float_eq(vm_pos.pop_stack().unwrap(), 4.0);

    // Negative (NaN) case
    let mut vm_neg = setup_vm_with_stack(&[-1.0]);
    (sqrt_native_def.func)(&mut vm_neg).unwrap();
    assert_float_eq(vm_neg.pop_stack().unwrap(), f64::NAN);
}

#[test]
fn test_native_is_nan() {
    let is_nan_native_def = &NATIVES[NATIVE_MAP["is_nan"].0 as usize];
    assert_eq!(is_nan_native_def.arity, 1);

    let mut vm_is_nan = setup_vm_with_stack(&[f64::NAN]);
    (is_nan_native_def.func)(&mut vm_is_nan).unwrap();
    assert_float_eq(vm_is_nan.pop_stack().unwrap(), 1.0);

    let mut vm_is_not_nan = setup_vm_with_stack(&[1.0]);
    (is_nan_native_def.func)(&mut vm_is_not_nan).unwrap();
    assert_float_eq(vm_is_not_nan.pop_stack().unwrap(), 0.0);
}

// --- Binary Pure Function Tests ---
#[test]
fn test_native_pow() {
    let mut vm = setup_vm_with_stack(&[2.0, 3.0]); // Stack: [base, exponent] -> pops exponent, then base
    let pow_native_def = &NATIVES[NATIVE_MAP["pow"].0 as usize];
    (pow_native_def.func)(&mut vm).unwrap();
    assert_eq!(pow_native_def.arity, 2);
    assert_float_eq(vm.pop_stack().unwrap(), 8.0);
}

#[test]
fn test_native_random_range() {
    let mut vm = setup_vm_with_stack(&[10.0, 20.0]); // Stack: [min, max]
    let random_range_def = &NATIVES[NATIVE_MAP["random_range"].0 as usize];
    (random_range_def.func)(&mut vm).unwrap();
    assert_eq!(random_range_def.arity, 2);
    let result = vm.pop_stack().unwrap();
    assert!(result >= 10.0 && result < 20.0);
}

// --- IO Native Tests ---
#[test]
fn test_native_print_success() {
    let mut vm = setup_vm_with_stack(&[]);
    let print_native_def = &NATIVES[NATIVE_MAP["print"].0 as usize];
    assert_eq!(print_native_def.arity, 1);

    // Create a list: (cons 72 (cons 105 nil)) -> "Hi"
    let list_end = vm.heap.register(HeapObject::Pair(105.0, NIL_VALUE));
    let list_start = vm.heap.register(HeapObject::Pair(72.0, encode_heap_pointer(list_end)));
    vm.stack.push(encode_heap_pointer(list_start));

    (print_native_def.func)(&mut vm).unwrap();
    // Print returns 1.0 on success
    assert_float_eq(vm.pop_stack().unwrap(), 1.0);
}

#[test]
fn test_native_print_error_non_list() {
    let mut vm = setup_vm_with_stack(&[42.0]); // Push a plain number, not a list pointer
    let print_native_def = &NATIVES[NATIVE_MAP["print"].0 as usize];

    let result = (print_native_def.func)(&mut vm);
    assert!(matches!(result, Err(InterpretError::Runtime(_))));
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("print expects a list"));
}