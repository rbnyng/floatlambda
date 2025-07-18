// tests/test_utils.rs

use float_lambda::{parse, Heap, Term};
use std::collections::HashMap;
use std::rc::Rc;

// Basic evaluation without prelude
pub fn eval_ok(input: &str) -> f64 {
    let term = parse(input).unwrap();
    let mut heap = Heap::new();
    term.eval(&Rc::new(HashMap::new()), &mut heap).unwrap()
}

// Evaluation that expects an error
#[allow(dead_code)]
pub fn eval_err(input: &str) -> bool {
    let term = parse(input);
    match term {
        Ok(t) => {
            let mut heap = Heap::new();
            t.eval(&Rc::new(HashMap::new()), &mut heap).is_err()
        }
        Err(_) => true,
    }
}

#[allow(dead_code)]
pub fn eval_with_timeout(input: &str, timeout_ms: u64) -> Result<f64, String> {
    let start = std::time::Instant::now();
    let result = eval_ok(input);
    let duration = start.elapsed();
    
    if duration.as_millis() > timeout_ms as u128 {
        Err(format!("Evaluation took too long: {}ms", duration.as_millis()))
    } else {
        Ok(result)
    }
}

// Create a heap and environment with prelude loaded
#[allow(dead_code)]
pub fn create_prelude_environment() -> (Heap, HashMap<String, f64>) {
    let mut heap = Heap::new();
    let mut global_env = HashMap::new();
    
    // Load prelude using the same logic as main.rs
    let prelude_code = include_str!("../src/prelude.fl");
    let term = parse(prelude_code).unwrap();
    
    // Walk the let chain and populate the environment
    let mut current_term = &term;
    loop {
        match current_term {
            Term::Let(name, value, body) => {
                let eval_env = Rc::new(global_env.clone());
                let value_val = value.eval(&eval_env, &mut heap).unwrap();
                global_env.insert(name.clone(), value_val);
                current_term = body;
            },
            Term::LetRec(name, value, body) => {
                let eval_env = Rc::new(global_env.clone());
                let get_func_term = Term::LetRec(name.clone(), value.clone(), Box::new(Term::Var(name.clone())));
                let func_val = get_func_term.eval(&eval_env, &mut heap).unwrap();
                global_env.insert(name.clone(), func_val);
                current_term = body;
            },
            _ => break,
        }
    }
    
    (heap, global_env)
}
// Evaluation with prelude loaded
#[allow(dead_code)]
pub fn eval_ok_with_prelude(input: &str) -> f64 {
    let (mut heap, global_env) = create_prelude_environment();
    let term = parse(input).unwrap();
    term.eval(&Rc::new(global_env), &mut heap).unwrap()
}

// Evaluation with prelude that expects an error
#[allow(dead_code)]
pub fn eval_err_with_prelude(input: &str) -> bool {
    let (mut heap, global_env) = create_prelude_environment();
    let term = parse(input);
    match term {
        Ok(t) => t.eval(&Rc::new(global_env), &mut heap).is_err(),
        Err(_) => true,
    }
}

// Helper to check if two lists are equal (for testing)
#[allow(dead_code)]
pub fn lists_equal(a: f64, b: f64, heap: &Heap) -> bool {
    let mut current_a = a;
    let mut current_b = b;
    
    loop {
        // Both nil
        if current_a == float_lambda::NIL_VALUE && current_b == float_lambda::NIL_VALUE {
            return true;
        }
        
        // One nil, one not
        if current_a == float_lambda::NIL_VALUE || current_b == float_lambda::NIL_VALUE {
            return false;
        }
        
        // Both should be pairs
        let (car_a, cdr_a) = if let Some(id) = float_lambda::memory::decode_heap_pointer(current_a) {
            if let Some(float_lambda::memory::HeapObject::Pair(car, cdr)) = heap.get(id) {
                (*car, *cdr)
            } else {
                return false;
            }
        } else {
            return false;
        };
        
        let (car_b, cdr_b) = if let Some(id) = float_lambda::memory::decode_heap_pointer(current_b) {
            if let Some(float_lambda::memory::HeapObject::Pair(car, cdr)) = heap.get(id) {
                (*car, *cdr)
            } else {
                return false;
            }
        } else {
            return false;
        };
        
        // Compare car values
        if (car_a - car_b).abs() > 1e-10 {
            return false;
        }
        
        // Move to next elements
        current_a = cdr_a;
        current_b = cdr_b;
    }
}

// Assertion helper for comparing floats with tolerance
#[allow(dead_code)]
pub fn assert_float_eq(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() < tolerance, 
            "Expected {}, got {}, difference: {}", 
            expected, actual, (actual - expected).abs());
}

// Assertion helper for comparing floats with default tolerance
#[allow(dead_code)]
pub fn assert_float_eq_default(actual: f64, expected: f64) {
    assert_float_eq(actual, expected, 1e-10);
}
