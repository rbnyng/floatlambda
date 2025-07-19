// src/vm/natives.rs

use crate::memory::{decode_heap_pointer, HeapObject, NIL_VALUE};
use crate::vm::ml_natives::ML_NATIVES;
use crate::vm::hof_natives::HOF_NATIVES;
use crate::vm::vm::{InterpretError, VM};
use lazy_static::lazy_static;
use std::collections::HashMap;

// Import the pure math functions
use crate::math;

// The signature for all native Rust functions callable by the VM.
pub type NativeFn = fn(&mut VM) -> Result<(), InterpretError>;

#[derive(Debug, Copy, Clone)]
pub struct NativeDef {
    pub name: &'static str,
    pub arity: usize,
    pub func: NativeFn,
}

// --- Special (Non-Standard) Native Functions ---

fn native_print(vm: &mut VM) -> Result<(), InterpretError> {
    let mut current_val = vm.pop_stack()?;
    loop {
        if current_val == NIL_VALUE { break; }
        let id = decode_heap_pointer(current_val)
            .ok_or_else(|| InterpretError::Runtime("print expects a list".to_string()))?;
        
        if let Some(HeapObject::Pair(car, cdr)) = vm.heap.get(id) {
            if let Some(c) = std::char::from_u32(*car as u32) { print!("{}", c); }
            current_val = *cdr;
        } else {
            return Err(InterpretError::Runtime("print expects a proper list".to_string()));
        }
    }
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    vm.stack.push(1.0);
    Ok(())
}

fn native_diff(vm: &mut VM) -> Result<(), InterpretError> {
    let x = vm.pop_stack()?;
    let func_val = vm.pop_stack()?;

    if decode_heap_pointer(func_val).is_none() {
        return Err(InterpretError::Runtime("'diff' expects a function as its first argument.".to_string()));
    }

    let h = 1e-6;

    vm.stack.push(func_val);
    vm.stack.push(x + h);
    vm.call_value(func_val, 1)?;
    let f_x_plus_h = vm.run()?;
    vm.pop_stack()?; 

    vm.stack.push(func_val);
    vm.stack.push(x - h);
    vm.call_value(func_val, 1)?;
    let f_x_minus_h = vm.run()?;
    vm.pop_stack()?;

    let derivative = (f_x_plus_h - f_x_minus_h) / (2.0 * h);
    vm.stack.push(derivative);
    Ok(())
}

fn native_integrate(vm: &mut VM) -> Result<(), InterpretError> {
    let b = vm.pop_stack()?;
    let a = vm.pop_stack()?;
    let func_val = vm.pop_stack()?;

    if decode_heap_pointer(func_val).is_none() {
        return Err(InterpretError::Runtime("'integrate' expects a function as its first argument.".to_string()));
    }

    let n = 1000;
    let h = (b - a) / (n as f64);
    let mut sum = 0.0;

    for i in 0..=n {
        let x = a + (i as f64) * h;
        
        vm.stack.push(func_val);
        vm.stack.push(x);
        vm.call_value(func_val, 1)?;
        let fx = vm.run()?;
        vm.pop_stack()?; 

        let weight = if i == 0 || i == n { 1.0 }
                     else if i % 2 == 1 { 4.0 }
                     else { 2.0 };
        sum += weight * fx;
    }
    
    let integral = sum * h / 3.0;
    vm.stack.push(integral);
    Ok(())
}

// A macro to simplify creating pure function wrappers.
macro_rules! pure_unary_fn {
    ($func:path) => {
        |vm: &mut VM| {
            let val = vm.pop_stack()?;
            vm.stack.push($func(val));
            Ok(())
        }
    };
}
macro_rules! pure_binary_fn {
    ($func:path) => {
        |vm: &mut VM| {
            let b = vm.pop_stack()?;
            let a = vm.pop_stack()?;
            vm.stack.push($func(a, b));
            Ok(())
        }
    };
}

// --- The Registry Definition ---

// Define the CORE natives separately.
const CORE_NATIVES: &[NativeDef] = &[
    // Special IO/VM functions
    NativeDef { name: "print", arity: 1, func: native_print },
    NativeDef { name: "diff", arity: 2, func: native_diff },
    NativeDef { name: "integrate", arity: 3, func: native_integrate },
    // Nullary constants
    NativeDef { name: "pi", arity: 0, func: |_vm| { _vm.stack.push(math::PI); Ok(()) } },
    NativeDef { name: "e", arity: 0, func: |_vm| { _vm.stack.push(math::E); Ok(()) } },
    NativeDef { name: "tau", arity: 0, func: |_vm| { _vm.stack.push(math::TAU); Ok(()) } },
    NativeDef { name: "sqrt2", arity: 0, func: |_vm| { _vm.stack.push(math::SQRT_2); Ok(()) } },
    NativeDef { name: "ln2", arity: 0, func: |_vm| { _vm.stack.push(math::LN_2); Ok(()) } },
    NativeDef { name: "ln10", arity: 0, func: |_vm| { _vm.stack.push(math::LN_10); Ok(()) } },
    // Nullary pure functions
    NativeDef { name: "random", arity: 0, func: |_vm| { _vm.stack.push(math::fl_random()); Ok(()) } },
    // Unary pure functions
    NativeDef { name: "sin", arity: 1, func: pure_unary_fn!(math::fl_sin) },
    NativeDef { name: "cos", arity: 1, func: pure_unary_fn!(math::fl_cos) },
    NativeDef { name: "tan", arity: 1, func: pure_unary_fn!(math::fl_tan) },
    NativeDef { name: "exp", arity: 1, func: pure_unary_fn!(math::fl_exp) },
    NativeDef { name: "log", arity: 1, func: pure_unary_fn!(math::fl_log) },
    NativeDef { name: "asin", arity: 1, func: pure_unary_fn!(math::fl_asin) },
    NativeDef { name: "acos", arity: 1, func: pure_unary_fn!(math::fl_acos) },
    NativeDef { name: "atan", arity: 1, func: pure_unary_fn!(math::fl_atan) },
    NativeDef { name: "sinh", arity: 1, func: pure_unary_fn!(math::fl_sinh) },
    NativeDef { name: "cosh", arity: 1, func: pure_unary_fn!(math::fl_cosh) },
    NativeDef { name: "tanh", arity: 1, func: pure_unary_fn!(math::fl_tanh) },
    NativeDef { name: "asinh", arity: 1, func: pure_unary_fn!(math::fl_asinh) },
    NativeDef { name: "acosh", arity: 1, func: pure_unary_fn!(math::fl_acosh) },
    NativeDef { name: "atanh", arity: 1, func: pure_unary_fn!(math::fl_atanh) },
    NativeDef { name: "sqrt", arity: 1, func: pure_unary_fn!(math::fl_sqrt) },
    NativeDef { name: "fuzzy_not", arity: 1, func: pure_unary_fn!(math::fl_fuzzy_not) },
    NativeDef { name: "cbrt", arity: 1, func: pure_unary_fn!(math::fl_cbrt) },
    NativeDef { name: "exp2", arity: 1, func: pure_unary_fn!(math::fl_exp2) },
    NativeDef { name: "log2", arity: 1, func: pure_unary_fn!(math::fl_log2) },
    NativeDef { name: "log10", arity: 1, func: pure_unary_fn!(math::fl_log10) },
    NativeDef { name: "floor", arity: 1, func: pure_unary_fn!(math::fl_floor) },
    NativeDef { name: "ceil", arity: 1, func: pure_unary_fn!(math::fl_ceil) },
    NativeDef { name: "round", arity: 1, func: pure_unary_fn!(math::fl_round) },
    NativeDef { name: "trunc", arity: 1, func: pure_unary_fn!(math::fl_trunc) },
    NativeDef { name: "fract", arity: 1, func: pure_unary_fn!(math::fl_fract) },
    NativeDef { name: "signum", arity: 1, func: pure_unary_fn!(math::fl_signum) },
    NativeDef { name: "gamma", arity: 1, func: pure_unary_fn!(math::fl_gamma) },
    NativeDef { name: "lgamma", arity: 1, func: pure_unary_fn!(math::fl_lgamma) },
    NativeDef { name: "erf", arity: 1, func: pure_unary_fn!(math::fl_erf) },
    NativeDef { name: "erfc", arity: 1, func: pure_unary_fn!(math::fl_erfc) },
    NativeDef { name: "degrees", arity: 1, func: pure_unary_fn!(math::fl_degrees) },
    NativeDef { name: "radians", arity: 1, func: pure_unary_fn!(math::fl_radians) },
    NativeDef { name: "is_nan", arity: 1, func: pure_unary_fn!(math::fl_is_nan) },
    NativeDef { name: "is_infinite", arity: 1, func: pure_unary_fn!(math::fl_is_infinite) },
    NativeDef { name: "is_finite", arity: 1, func: pure_unary_fn!(math::fl_is_finite) },
    NativeDef { name: "is_normal", arity: 1, func: pure_unary_fn!(math::fl_is_normal) },
    NativeDef { name: "abs", arity: 1, func: pure_unary_fn!(f64::abs) },
    // Binary pure functions
    NativeDef { name: "pow", arity: 2, func: pure_binary_fn!(math::fl_pow) },
    NativeDef { name: "atan2", arity: 2, func: pure_binary_fn!(math::fl_atan2) },
    NativeDef { name: "random_range", arity: 2, func: pure_binary_fn!(math::fl_random_range) },
    NativeDef { name: "random_normal", arity: 2, func: pure_binary_fn!(math::fl_random_normal) },
    NativeDef { name: "hypot", arity: 2, func: pure_binary_fn!(math::fl_hypot) },
    NativeDef { name: "copysign", arity: 2, func: pure_binary_fn!(math::fl_copysign) },
    NativeDef { name: "min", arity: 2, func: pure_binary_fn!(f64::min) },
    NativeDef { name: "max", arity: 2, func: pure_binary_fn!(f64::max) },
    NativeDef { name: "fuzzy_and", arity: 2, func: pure_binary_fn!(math::fl_fuzzy_and) },
    NativeDef { name: "fuzzy_or", arity: 2, func: pure_binary_fn!(math::fl_fuzzy_or) },
];

lazy_static! {
    // Combine the core and ML natives into a single Vec for the VM to use.
    // The VM will use the index in this final Vec for dispatching OpNative.
    pub static ref NATIVES: Vec<NativeDef> = {
        let mut all_natives = Vec::new();
        all_natives.extend_from_slice(CORE_NATIVES);
        all_natives.extend_from_slice(HOF_NATIVES);
        all_natives.extend_from_slice(ML_NATIVES);
        all_natives
    };

    // Create a unified map for the compiler. This is built once at runtime
    // from the unified NATIVES list, ensuring indices are always correct.
    pub static ref NATIVE_MAP: HashMap<&'static str, (u8, usize)> = {
        let mut map = HashMap::new();
        for (i, native) in NATIVES.iter().enumerate() {
            map.insert(native.name, (i as u8, native.arity));
        }
        map
    };
}