// src/vm/natives.rs

use crate::memory::{decode_heap_pointer, HeapObject, NIL_VALUE};
use crate::vm::vm::{InterpretError, VM};
use paste::paste;

// Import the pure math functions
use crate::math;

// The signature for all native Rust functions callable by the VM.
pub type NativeFn = fn(&mut VM) -> Result<(), InterpretError>;

pub struct NativeDef {
    pub name: &'static str,
    pub arity: usize,
    pub func: NativeFn,
}

// --- Macro Template System ---

macro_rules! define_native_functions {
    (
        // Special-case functions (like print) that need direct VM access
        io_natives: [$(($io_name:literal, $io_index:expr, $io_arity:expr, $io_func:ident)),*],
        // Nullary functions that return a fixed constant
        nullary_constant: [$(($n_const_name:literal, $n_const_index:expr, $n_const_val:path)),*],
        // Nullary functions that call a pure function (e.g., random)
        nullary_pure_fn: [$(($n_pure_name:literal, $n_pure_index:expr, $n_pure_func:path)),*],
        // Unary functions that call a pure function
        unary_pure_fn: [$(($u_pure_name:literal, $u_pure_index:expr, $u_pure_func:path)),*],
        // Binary functions that call a pure function
        binary_pure_fn: [$(($b_pure_name:literal, $b_pure_index:expr, $b_pure_func:path)),*]
    ) => {
        paste! {
            // 1. Generate wrappers for all NULLARY PURE functions.
            $(
                fn [<native_ $n_pure_name>](vm: &mut VM) -> Result<(), InterpretError> {
                    vm.stack.push($n_pure_func());
                    Ok(())
                }
            )*

            // 2. Generate wrappers for all UNARY PURE functions.
            $(
                fn [<native_ $u_pure_name>](vm: &mut VM) -> Result<(), InterpretError> {
                    let val = vm.pop_stack()?;
                    vm.stack.push($u_pure_func(val));
                    Ok(())
                }
            )*

            // 3. Generate wrappers for all BINARY PURE functions.
            $(
                fn [<native_ $b_pure_name>](vm: &mut VM) -> Result<(), InterpretError> {
                    let b = vm.pop_stack()?;
                    let a = vm.pop_stack()?;
                    vm.stack.push($b_pure_func(a, b));
                    Ok(())
                }
            )*
        }

        // 4. Define the static NATIVES array for the VM's runtime.
        pub static NATIVES: &[NativeDef] = &[
            // Special IO natives
            $(NativeDef { name: $io_name, arity: $io_arity, func: $io_func }),*,
            // Nullary constants
            $(NativeDef { name: $n_const_name, arity: 0, func: |_vm| { _vm.stack.push($n_const_val); Ok(()) } }),*,
            // Nullary pure functions
            $(paste! { NativeDef { name: $n_pure_name, arity: 0, func: [<native_ $n_pure_name>] } }),*,
            // Unary pure functions
            $(paste! { NativeDef { name: $u_pure_name, arity: 1, func: [<native_ $u_pure_name>] } }),*,
            // Binary pure functions
            $(paste! { NativeDef { name: $b_pure_name, arity: 2, func: [<native_ $b_pure_name>] } }),*,
        ];

        // 5. Define the static NATIVE_MAP for the compiler's lookup.
        // This maps the string name to its index in NATIVES and its arity.
        pub static NATIVE_MAP: phf::Map<&'static str, (u8, usize)> = phf::phf_map! {
            $( $io_name => ($io_index, $io_arity), )*
            $( $n_const_name => ($n_const_index, 0), )*
            $( $n_pure_name => ($n_pure_index, 0), )*
            $( $u_pure_name => ($u_pure_index, 1), )*
            $( $b_pure_name => ($b_pure_index, 2), )*
        };
    };
}

// --- Special (Non-Standard) Native Functions ---

// Keep print separate as it directly interacts with VM heap and IO
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

// --- The Registry Definition ---

// Now, we use our macro to declare all functions.
// Assign unique indices to each native function/constant.
define_native_functions! {
    io_natives: [
        // name,    index, arity, function_name
        ("print",  0,     1,     native_print)
    ],
    nullary_constant: [
        // name,    index, constant_path
        ("pi",     1,     math::PI),
        ("e",      2,     math::E),
        ("tau",    3,     math::TAU),
        ("sqrt2",  4,     math::SQRT_2),
        ("ln2",    5,     math::LN_2),
        ("ln10",   6,     math::LN_10)
    ],
    nullary_pure_fn: [
        // name,    index, pure_function_path
        ("random", 7,     math::fl_random)
    ],
    unary_pure_fn: [
        // name,    index, pure_function_path
        ("sin",    8,     math::fl_sin),
        ("cos",    9,     math::fl_cos),
        ("tan",    10,    math::fl_tan),
        ("exp",    11,    math::fl_exp),
        ("log",    12,    math::fl_log),
        ("asin",   13,    math::fl_asin),
        ("acos",   14,    math::fl_acos),
        ("atan",   15,    math::fl_atan),
        ("sinh",   16,    math::fl_sinh),
        ("cosh",   17,    math::fl_cosh),
        ("tanh",   18,    math::fl_tanh),
        ("asinh",  19,    math::fl_asinh),
        ("acosh",  20,    math::fl_acosh),
        ("atanh",  21,    math::fl_atanh),
        ("sqrt",   22,    math::fl_sqrt),
        ("cbrt",   23,    math::fl_cbrt),
        ("exp2",   24,    math::fl_exp2),
        ("log2",   25,    math::fl_log2),
        ("log10",  26,    math::fl_log10),
        ("floor",  27,    math::fl_floor),
        ("ceil",   28,    math::fl_ceil),
        ("round",  29,    math::fl_round),
        ("trunc",  30,    math::fl_trunc),
        ("fract",  31,    math::fl_fract),
        ("signum", 32,    math::fl_signum),
        ("gamma",  33,    math::fl_gamma),
        ("lgamma", 34,    math::fl_lgamma),
        ("erf",    35,    math::fl_erf),
        ("erfc",   36,    math::fl_erfc),
        ("degrees",37,    math::fl_degrees),
        ("radians",38,    math::fl_radians),
        ("is_nan", 39,    math::fl_is_nan),
        ("is_infinite", 40, math::fl_is_infinite),
        ("is_finite", 41, math::fl_is_finite),
        ("is_normal", 42, math::fl_is_normal)
    ],
    binary_pure_fn: [
        // name,    index, pure_function_path
        ("pow",    43,    math::fl_pow),
        ("atan2",  44,    math::fl_atan2),
        ("random_range", 45, math::fl_random_range),
        ("random_normal", 46, math::fl_random_normal), // Updated to binary arity
        ("hypot",  47,    math::fl_hypot),
        ("copysign", 48,  math::fl_copysign)
        // Note: min/max are VM opcodes (+-*/ <>=) handled directly, not natives.
    ]
}
