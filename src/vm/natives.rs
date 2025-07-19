// src/vm/natives.rs

use crate::memory::{decode_heap_pointer, HeapObject, NIL_VALUE};
use crate::vm::vm::{InterpretError, VM};
use paste::paste;

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
        // A list of special-case functions defined manually.
        specials: [$(($s_name:literal, $s_index:expr, $s_arity:expr, $s_func:ident)),*],
        // A list of standard unary f64->f64 math functions.
        unary: [$(($u_name:literal, $u_index:expr, $u_func:path)),*],
        // A list of standard binary (f64, f64)->f64 math functions.
        binary: [$(($b_name:literal, $b_index:expr, $b_func:path)),*]
    ) => {
        // Use paste to generate unique function names like `native_sin`, `native_cos`, etc.
        paste! {
            // 1. Generate wrappers for all UNARY functions.
            $(
                fn [<native_ $u_name>](vm: &mut VM) -> Result<(), InterpretError> {
                    let val = vm.pop_stack()?;
                    vm.stack.push($u_func(val));
                    Ok(())
                }
            )*

            // 2. Generate wrappers for all BINARY functions.
            $(
                fn [<native_ $b_name>](vm: &mut VM) -> Result<(), InterpretError> {
                    let b = vm.pop_stack()?;
                    let a = vm.pop_stack()?;
                    vm.stack.push($b_func(a, b));
                    Ok(())
                }
            )*
        }

        // 3. Define the static NATIVES array for the VM's runtime.
        pub static NATIVES: &[NativeDef] = &[
            $(NativeDef { name: $s_name, arity: $s_arity, func: $s_func }),*,
            $(paste! { NativeDef { name: $u_name, arity: 1, func: [<native_ $u_name>] } }),*,
            $(paste! { NativeDef { name: $b_name, arity: 2, func: [<native_ $b_name>] } }),*,
        ];

        // 4. Define the static NATIVE_MAP for the compiler's lookup.
        pub static NATIVE_MAP: phf::Map<&'static str, (u8, usize)> = phf::phf_map! {
            $( $s_name => ($s_index, $s_arity), )*
            $( $u_name => ($u_index, 1), )*
            $( $b_name => ($b_index, 2), )*
        };
    };
}

// --- Special (Non-Standard) Native Functions ---

fn native_sqrt(vm: &mut VM) -> Result<(), InterpretError> {
    let val = vm.pop_stack()?;
    let result = if val < 0.0 { f64::NAN } else { val.sqrt() };
    vm.stack.push(result);
    Ok(())
}

fn native_pi(vm: &mut VM) -> Result<(), InterpretError> {
    vm.stack.push(std::f64::consts::PI);
    Ok(())
}

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
// To add more, just add a line to the appropriate list.
define_native_functions! {
    specials: [
        // name,    index, arity, function_name
        ("sqrt",   0,     1,     native_sqrt),
        ("pi",     1,     0,     native_pi),
        ("print",  2,     1,     native_print)
    ],
    unary: [
        // name,    index, rust_function_path
        ("sin",    3,     f64::sin),
        ("cos",    4,     f64::cos),
        ("tan",    5,     f64::tan),
        ("exp",    6,     f64::exp),
        ("log",    7,     f64::ln), // Note: FloatLambda `log` is natural log
        ("abs",    8,     f64::abs),
        ("ceil",   9,     f64::ceil),
        ("floor",  10,    f64::floor)
    ],
    binary: [
        // name,    index, rust_function_path
        ("pow",    11,    f64::powf),
        ("atan2",  12,    f64::atan2),
        ("min",    13,    f64::min),
        ("max",    14,    f64::max)
    ]
}