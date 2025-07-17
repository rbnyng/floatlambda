// src/evaluator.rs

use std::rc::Rc;
use rand::Rng;

use crate::ast::Term;
use crate::error::EvalError;
use crate::memory::{
    Environment, Heap, Closure, BuiltinClosure, HeapObject,
    NIL_VALUE, encode_heap_pointer, decode_heap_pointer
};

/// A helper function to apply a FloatLambda function (represented by its f64 value)
/// to an argument. This is used by Term::App and our new list built-ins.
pub fn apply_function(func_val: f64, arg_val: f64, heap: &mut Heap) -> Result<f64, EvalError> {
    if let Some(id) = decode_heap_pointer(func_val) {
        // Clone the object to avoid mutable borrow conflicts with the heap.
        let heap_obj = heap.get(id).cloned().ok_or(EvalError::DanglingPointerError(id))?;

        match heap_obj {
            HeapObject::UserFunc(closure) => {
                let mut new_env_map = closure.env.as_ref().clone(); 
                new_env_map.insert(closure.param, arg_val);
                closure.body.eval(&Rc::new(new_env_map), heap)
            }

            HeapObject::BuiltinFunc(mut closure) => {
                closure.args.push(arg_val);
                if closure.args.len() == closure.arity {
                    // Pass the heap to builtins that might need it (like cons).
                    execute_builtin(&closure.op, &closure.args, heap)
                } else {
                    let new_id = heap.register(HeapObject::BuiltinFunc(closure));
                    Ok(encode_heap_pointer(new_id))
                }
            }
            // Applying a pair is a type error.
            HeapObject::Pair(_, _) => {
                Err(EvalError::TypeError(format!("Cannot apply a non-function value: Pair<{}>", id)))
            }
            HeapObject::Free(_) => {
                Err(EvalError::DanglingPointerError(id))
            }
        }
    } else {
         // Check if it's nil before declaring it a plain number
        if func_val == NIL_VALUE {
            Err(EvalError::TypeError("Cannot apply a non-function value: nil".to_string()))
        } else {
            Err(EvalError::TypeError(format!("Cannot apply a non-function value: {}", func_val)))
        }
    }
}


// --- The Evaluator ---
impl Term {
    pub fn eval(&self, env: &Environment, heap: &mut Heap) -> Result<f64, EvalError> {
        match self {
            Term::Float(n) => Ok(*n),
            Term::Nil => Ok(NIL_VALUE),

            Term::Var(name) => env
                .get(name)
                .copied()
                .ok_or_else(|| EvalError::UnboundVariable(name.clone())),

            Term::Lam(param, body) => {
                let closure = Closure {
                    param: param.clone(),
                    body: body.clone(),
                    env: env.clone(),
                };
                let id = heap.register(HeapObject::UserFunc(closure));
                Ok(encode_heap_pointer(id))
            }
            
            Term::Let(name, value, body) => {
                let value_val = value.eval(env, heap)?;
                let mut new_env_map = env.as_ref().clone(); 
                new_env_map.insert(name.clone(), value_val);
                body.eval(&Rc::new(new_env_map), heap)
            }

            Term::LetRec(name, value, body) => {
                let mut temp_env_map = env.as_ref().clone();
                temp_env_map.insert(name.clone(), f64::NAN); // Insert placeholder
                let temp_env = Rc::new(temp_env_map);
                
                let value_val = value.eval(&temp_env, heap)?;

                if decode_heap_pointer(value_val).is_none() {
                    return Err(EvalError::TypeError(
                        "let rec expression must result in a heap-allocated value (a function or pair)".to_string()
                    ));
                }

                fn patch_recursively(val: f64, name: &str, final_val: f64, heap: &mut Heap) {
                    let mut pair_to_trace: Option<(f64, f64)> = None;
                
                    if let Some(id) = decode_heap_pointer(val) {
                        if let Some(obj) = heap.get_mut(id) {
                            match obj {
                                HeapObject::UserFunc(closure) => {
                                    if let Some(map) = Rc::get_mut(&mut closure.env) {
                                        if map.get(name).map_or(false, |v| v.is_nan()) {
                                            map.insert(name.to_string(), final_val);
                                        }
                                    } else {
                                        let mut new_map = closure.env.as_ref().clone();
                                        if new_map.get(name).map_or(false, |v| v.is_nan()) {
                                            new_map.insert(name.to_string(), final_val);
                                            closure.env = Rc::new(new_map);
                                        }
                                    }
                                },
                                HeapObject::Pair(car, cdr) => {
                                    pair_to_trace = Some((*car, *cdr));
                                },
                                _ => {}
                            }
                        }
                    }
                
                    if let Some((car, cdr)) = pair_to_trace {
                        patch_recursively(car, name, final_val, heap);
                        patch_recursively(cdr, name, final_val, heap);
                    }
                }

                patch_recursively(value_val, &name, value_val, heap);
            
                let mut body_env_map = env.as_ref().clone();
                body_env_map.insert(name.clone(), value_val);
                body.eval(&Rc::new(body_env_map), heap)
            }

            Term::Builtin(op) => {
                let arity = get_builtin_arity(op)?;
                if arity == 0 {
                    return execute_builtin(op, &[], heap);
                }

                let builtin_closure = BuiltinClosure {
                    op: op.clone(),
                    arity,
                    args: Vec::new(),
                };
                let id = heap.register(HeapObject::BuiltinFunc(builtin_closure));
                Ok(encode_heap_pointer(id))
            }

            Term::App(func, arg) => {
                let func_val = func.eval(env, heap)?;
                let arg_val = arg.eval(env, heap)?;
                // Refactored logic into the helper function
                apply_function(func_val, arg_val, heap)
            }
            
            Term::If(cond, then_branch, else_branch) => {
                let condition_val = cond.eval(env, heap)?;

                if condition_val == 0.0 || condition_val == NIL_VALUE {
                    return else_branch.eval(env, heap);
                }
                if condition_val == 1.0 {
                    return then_branch.eval(env, heap);
                }

                let then_val = then_branch.eval(env, heap)?;
                let else_val = else_branch.eval(env, heap)?;
                let weight = condition_val.max(0.0).min(1.0);
                Ok(weight * then_val + (1.0 - weight) * else_val)
            }
        }
    }
}

// --- Builtin Logic Helpers ---
fn get_builtin_arity(op: &str) -> Result<usize, EvalError> {
    match op {
        // Nullary
        "read-char" | "read-line" => Ok(0),
        // Unary
        "neg" | "abs" | "sqrt" | "fuzzy_not" | "car" | "cdr" | "print" | "length" => Ok(1),
        // Binary
        "+" | "-" | "*" | "/" | "==" | "eq?" | "<" | ">" | "<=" | ">=" | "min" | "max" |
        "cons" | "fuzzy_and" | "fuzzy_or" | "rem" | "div" | "map" | "filter" => Ok(2),
        // Ternary
        "foldl" => Ok(3),
        _ => Err(EvalError::TypeError(format!("Unknown builtin: {}", op))),
    }
}

fn execute_builtin(op: &str, args: &[f64], heap: &mut Heap) -> Result<f64, EvalError> {
    match op {
        // --- HIGHER-ORDER BUILTINS ---
        "length" => {
            let mut count = 0.0;
            let mut current_list = args[0];
            loop {
                if current_list == NIL_VALUE { break Ok(count); }
                
                if let Some(id) = decode_heap_pointer(current_list) {
                    if let Some(HeapObject::Pair(_, cdr)) = heap.get(id) {
                        count += 1.0;
                        current_list = *cdr;
                    } else {
                        break Err(EvalError::TypeError("Argument to 'length' must be a proper list.".to_string()));
                    }
                } else {
                    break Err(EvalError::TypeError("Argument to 'length' must be a proper list.".to_string()));
                }
            }
        }
        "map" => {
            let func = args[0];
            let mut current_list = args[1];
            let mut results = Vec::new();

            loop {
                if current_list == NIL_VALUE { break; }

                let (car, cdr) = if let Some(id) = decode_heap_pointer(current_list) {
                     if let Some(HeapObject::Pair(car, cdr)) = heap.get(id) {
                        (*car, *cdr)
                    } else {
                        return Err(EvalError::TypeError("Second argument to 'map' must be a proper list.".to_string()));
                    }
                } else {
                     return Err(EvalError::TypeError("Second argument to 'map' must be a proper list.".to_string()));
                };
                
                let mapped_val = apply_function(func, car, heap)?;
                results.push(mapped_val);
                current_list = cdr;
            }
            
            // Build the new list from results
            let mut new_list = NIL_VALUE;
            for val in results.iter().rev() {
                let new_pair = HeapObject::Pair(*val, new_list);
                let new_id = heap.register(new_pair);
                new_list = encode_heap_pointer(new_id);
            }
            Ok(new_list)
        }
        "filter" => {
            let predicate = args[0];
            let mut current_list = args[1];
            let mut results = Vec::new();

            loop {
                if current_list == NIL_VALUE { break; }

                let (car, cdr) = if let Some(id) = decode_heap_pointer(current_list) {
                     if let Some(HeapObject::Pair(car, cdr)) = heap.get(id) {
                        (*car, *cdr)
                    } else {
                        return Err(EvalError::TypeError("Second argument to 'filter' must be a proper list.".to_string()));
                    }
                } else {
                     return Err(EvalError::TypeError("Second argument to 'filter' must be a proper list.".to_string()));
                };

                let should_keep = apply_function(predicate, car, heap)?;
                // Keep if result is not 0.0 or nil
                if should_keep != 0.0 && should_keep != NIL_VALUE {
                    results.push(car);
                }
                current_list = cdr;
            }
            
            let mut new_list = NIL_VALUE;
            for val in results.iter().rev() {
                let new_pair = HeapObject::Pair(*val, new_list);
                let new_id = heap.register(new_pair);
                new_list = encode_heap_pointer(new_id);
            }
            Ok(new_list)
        }
        "foldl" => {
            let func = args[0];
            let mut acc = args[1];
            let mut current_list = args[2];
            
            loop {
                 if current_list == NIL_VALUE { break Ok(acc); }

                 let (car, cdr) = if let Some(id) = decode_heap_pointer(current_list) {
                     if let Some(HeapObject::Pair(car, cdr)) = heap.get(id) {
                        (*car, *cdr)
                    } else {
                        break Err(EvalError::TypeError("Third argument to 'foldl' must be a proper list.".to_string()));
                    }
                } else {
                    break Err(EvalError::TypeError("Third argument to 'foldl' must be a proper list.".to_string()));
                };

                let partial_app = apply_function(func, acc, heap)?;
                acc = apply_function(partial_app, car, heap)?;
                current_list = cdr;
            }
        }

        // Standard arithmetic/logic (no heap access)
        "neg" => Ok(-args[0]),
        "abs" => Ok(args[0].abs()),
        "sqrt" => Ok(if args[0] < 0.0 { f64::NAN } else { args[0].sqrt() }),
        "fuzzy_not" => Ok(1.0 - args[0].max(0.0).min(1.0)),
        "+" => Ok(args[0] + args[1]),
        "-" => Ok(args[0] - args[1]),
        "*" => Ok(args[0] * args[1]),
        "/" => Ok(if args[1] == 0.0 { f64::INFINITY } else { args[0] / args[1] }),
        "<" => Ok(if args[0] < args[1] { 1.0 } else { 0.0 }),
        ">" => Ok(if args[0] > args[1] { 1.0 } else { 0.0 }),
        "<=" => Ok(if args[0] <= args[1] { 1.0 } else { 0.0 }),
        ">=" => Ok(if args[0] >= args[1] { 1.0 } else { 0.0 }),
        "min" => Ok(args[0].min(args[1])),
        "max" => Ok(args[0].max(args[1])),
        "fuzzy_and" => Ok(args[0] * args[1]),
        "fuzzy_or" => Ok(args[0] + args[1] - (args[0] * args[1])),
        "div" => Ok((args[0] / args[1]).floor()),
        "rem" => Ok(args[0] % args[1]),
        "eq?" => {
            // eq? is for strict, physical equality.
            let p1 = decode_heap_pointer(args[0]);
            let p2 = decode_heap_pointer(args[1]);

            let result = if p1.is_some() || p2.is_some() {
                // Pointer equality for heap objects.
                p1 == p2 
            } else {
                // Strict bit-wise float equality for numbers and nil.
                // This correctly handles 0.0 vs -0.0 and NaNs.
                args[0].to_bits() == args[1].to_bits()
            };
            Ok(if result { 1.0 } else { 0.0 })
        },
        // '==' is always fuzzy. Note this will produce NaN for pointers/nil.
        "==" => {
            Ok(fuzzy_eq(args[0], args[1]))
        },
        
        // --- List primitives that interact with the heap ---
        "cons" => {
            let obj = HeapObject::Pair(args[0], args[1]);
            let id = heap.register(obj);
            Ok(encode_heap_pointer(id))
        }
        "car" => {
            if let Some(id) = decode_heap_pointer(args[0]) {
                match heap.get(id) {
                    Some(HeapObject::Pair(car, _)) => Ok(*car),
                    _ => Err(EvalError::TypeError(format!("'car' expects a pair, but got another heap object."))),
                }
            } else {
                Err(EvalError::TypeError(format!("'car' expects a pair, but got a number or nil.")))
            }
        }
        "cdr" => {
            if let Some(id) = decode_heap_pointer(args[0]) {
                match heap.get(id) {
                    Some(HeapObject::Pair(_, cdr)) => Ok(*cdr),
                    _ => Err(EvalError::TypeError(format!("'cdr' expects a pair, but got another heap object."))),
                }
            } else {
                Err(EvalError::TypeError(format!("'cdr' expects a pair, but got a number or nil.")))
            }
        }

        // --- I/O Builtins ---

        "print" => {
            let mut current_val = args[0];
            let mut rng = rand::thread_rng();

            // Loop through the list (cons cells)
            loop {
                if current_val == NIL_VALUE {
                    break;
                }

                let id = decode_heap_pointer(current_val).ok_or_else(|| {
                    EvalError::TypeError("print expects a list, but found a number".to_string())
                })?;
                
                if let Some(HeapObject::Pair(car, cdr)) = heap.get(id) {
                    let char_f64 = *car;
                    let next_cdr = *cdr;

                    // --- Probabilistic Character Logic ---
                    if char_f64.is_finite() {
                        let floor_code = char_f64.floor() as u32;
                        let fract = char_f64.fract();
                        
                        let final_code = if fract > 0.0 && rng.gen::<f64>() < fract {
                            char_f64.ceil() as u32
                        } else {
                            floor_code
                        };

                        if let Some(c) = std::char::from_u32(final_code) {
                            print!("{}", c);
                        }
                    }
                    // --- End Probabilistic Logic ---
                    
                    current_val = next_cdr;
                } else {
                    return Err(EvalError::TypeError("print expects a proper list".to_string()));
                }
            }
            
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            Ok(1.0) // Return 1.0 for success
        }

        "read-char" => {
            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_err() {
                return Ok(NIL_VALUE); // Return nil on error
            }
            // Return the code of the first character, or 0 if the line is empty
            let code = input.chars().next().map_or(0, |c| c as u32);
            Ok(code as f64)
        }
        
        "read-line" => {
            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_err() {
                    return Ok(NIL_VALUE);
            }
            
            // Build the list of character codes in reverse
            let mut list_val = NIL_VALUE;
            for c in input.trim_end().chars().rev() {
                let char_code = c as u32 as f64;
                let new_pair = HeapObject::Pair(char_code, list_val);
                let new_id = heap.register(new_pair);
                list_val = encode_heap_pointer(new_id);
            }
            Ok(list_val)
        }
        
        _ => unreachable!(),
    }
}

pub fn fuzzy_eq(x: f64, y: f64) -> f64 {
    let scale_factor = 1.0;
    (-((x - y).abs() / scale_factor)).exp()
}