// src/math.rs

use crate::error::EvalError;

// This module will house stateless, pure math functions.

pub fn execute_math_builtin(op: &str, args: &[f64]) -> Result<f64, EvalError> {
    match op {
        "sin" => Ok(args[0].sin()),
        "cos" => Ok(args[0].cos()),
        "exp" => Ok(args[0].exp()),
        "log" => {
            if args[0] <= 0.0 {
                Ok(f64::NAN) // Log is undefined for non-positive numbers
            } else {
                Ok(args[0].ln())
            }
        }
        _ => Err(EvalError::TypeError(format!("Unknown math builtin: {}", op))),
    }
}

pub fn get_math_builtin_arity(op: &str) -> Option<usize> {
    match op {
        "sin" | "cos" | "exp" | "log" => Some(1),
        _ => None,
    }
}