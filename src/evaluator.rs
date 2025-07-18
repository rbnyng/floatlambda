// src/evaluator.rs

use std::rc::Rc;
use rand::Rng;

use crate::ast::Term;
use crate::error::EvalError;
use crate::memory::{
    Environment, Heap, ASTClosure, BuiltinClosure, HeapObject,
    NIL_VALUE, encode_heap_pointer, decode_heap_pointer
};
use crate::ml::{self}; 
use crate::math; 

// A helper function to apply a FloatLambda function (represented by its f64 value)
// to an argument. This is used by Term::App and our new list built-ins.
pub fn apply_function(func_val: f64, arg_val: f64, heap: &mut Heap) -> Result<f64, EvalError> {

    // println!("apply_function called with func_val={}, arg_val={}", func_val, arg_val);

    if let Some(id) = decode_heap_pointer(func_val) {
        // println!("Applying heap object ID {}", id);
        // Clone the object to avoid mutable borrow conflicts with the heap.
        let heap_obj = heap.get(id).cloned().ok_or(EvalError::DanglingPointerError(id))?;

        match heap_obj {
            HeapObject::UserFunc(closure) => {
                                // println!("Applying UserFunc");
                                let mut new_env_map = closure.env.as_ref().clone(); 
                                new_env_map.insert(closure.param, arg_val);
                                closure.body.eval(&Rc::new(new_env_map), heap)
                            }
            HeapObject::BuiltinFunc(mut closure) => {
                                // println!("Applying BuiltinFunc: op={}, current_args={:?}, arity={}", 
                                // closure.op, closure.args, closure.arity);

                                closure.args.push(arg_val);
                                if closure.args.len() == closure.arity {
                                    // Pass the heap to builtins that might need it (like cons).
                                    // println!("Executing builtin {} with args {:?}", closure.op, closure.args);
                                    execute_builtin(&closure.op, &closure.args, heap)
                                } else {
                                    // println!("Creating partial application");
                                    let new_id = heap.register(HeapObject::BuiltinFunc(closure));
                                    Ok(encode_heap_pointer(new_id))
                                }
                            }
            HeapObject::Pair(_, _) => {
                                Err(EvalError::TypeError(format!("Cannot apply a non-function value: Pair<{}>", id)))
                            }
            HeapObject::Tensor(_) => {
                                Err(EvalError::TypeError(format!("Cannot apply a non-function value: Tensor<{}>", id)))
                            }
            HeapObject::Free(_) => {
                                Err(EvalError::DanglingPointerError(id))
                            }
            HeapObject::Function(_function) => Err(EvalError::TypeError("Cannot apply a VM function in the tree-walker.".to_string())),
            HeapObject::Closure(_) => Err(EvalError::TypeError("Cannot apply a VM closure in the tree-walker.".to_string())),
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


// --- CPS Trampoline Helpers ---
#[derive(Debug)]
enum Continuation {
    // Sentinel value indicating the end of the computation.
    Done,
    // Finished evaluating the value of a let, now evaluate the body.
    Let { name: String, body: Box<Term>, env: Environment },
    // Finished evaluating the value of a let rec, now evaluate the body.
    LetRec { name: String, body: Box<Term>, env: Environment },
    // Finished evaluating the function part of an App, now evaluate the argument.
    AppFunc { arg: Box<Term>, env: Environment },
    // Finished evaluating the argument part of an App, now apply the function to the argument.
    AppArg { func_val: f64 },
    // Finished evaluating the cond of an If, now evaluate the branches.
    IfCond { then_branch: Box<Term>, else_branch: Box<Term>, env: Environment },
    // (For fuzzy If) Finished then branch, now evaluate else branch.
    IfThen { else_branch: Box<Term>, env: Environment, cond_val: f64 },
    // (For fuzzy If) Finished else branch, now compute the final blend.
    IfElse { then_val: f64, cond_val: f64 },
}

// Helper for let rec to patch a closure's environment with its own pointer.
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


// --- The Evaluator ---
impl Term {
    // Evaluates a Term using a trampoline to implement Continuation-Passing Style (CPS).
    // This version uses a two-phase loop to correctly manage evaluation and continuation.
    pub fn eval(&self, env: &Environment, heap: &mut Heap) -> Result<f64, EvalError> {
        let mut current_term = self.clone();
        let mut current_env = env.clone();
        let mut cont_stack: Vec<Continuation> = vec![Continuation::Done];
        let mut result_val: f64;

        'trampoline: loop {
            // --- PHASE 1: EVALUATION ---
            // Keep deconstructing terms until we hit a base case that produces a value.
            match &current_term {
                Term::Float(n) => result_val = *n,
                Term::Nil => result_val = NIL_VALUE,
                Term::Var(name) => {
                    result_val = current_env.get(name)
                        .copied()
                        .ok_or_else(|| EvalError::UnboundVariable(name.clone()))?;
                }
                Term::Lam(param, body) => {
                    let closure = ASTClosure { param: param.clone(), body: body.clone(), env: current_env.clone() };
                    let id = heap.register(HeapObject::UserFunc(closure));
                    result_val = encode_heap_pointer(id);
                }
                Term::Builtin(op) => {
                    let arity = get_builtin_arity(op)?;
                    if arity == 0 {
                        result_val = execute_builtin(op, &[], heap)?;
                    } else {
                        let builtin_closure = BuiltinClosure { op: op.clone(), arity, args: Vec::new() };
                        let id = heap.register(HeapObject::BuiltinFunc(builtin_closure));
                        result_val = encode_heap_pointer(id);
                    }
                }
                Term::Let(name, value, body) => {
                    cont_stack.push(Continuation::Let { name: name.clone(), body: body.clone(), env: current_env.clone() });
                    current_term = (**value).clone();
                    // The value is evaluated in the current environment, which is preserved.
                    continue 'trampoline;
                }
                Term::LetRec(name, value, body) => {
                    let mut temp_env_map = current_env.as_ref().clone();
                    temp_env_map.insert(name.clone(), f64::NAN);
                    let temp_env = Rc::new(temp_env_map);
                    
                    cont_stack.push(Continuation::LetRec { name: name.clone(), body: body.clone(), env: current_env.clone() });
                    current_term = (**value).clone();
                    current_env = temp_env;
                    continue 'trampoline;
                }
                Term::App(func, arg) => {
                    cont_stack.push(Continuation::AppFunc { arg: arg.clone(), env: current_env.clone() });
                    current_term = (**func).clone();
                    continue 'trampoline;
                }
                Term::If(cond, then_branch, else_branch) => {
                    cont_stack.push(Continuation::IfCond { then_branch: then_branch.clone(), else_branch: else_branch.clone(), env: current_env.clone() });
                    current_term = (**cond).clone();
                    continue 'trampoline;
                }
            };

            // --- PHASE 2: CONTINUATION ---
            // A value has been produced in result_val. Now, consume continuations
            // until we need to evaluate a new term.
            loop {
                match cont_stack.pop().unwrap() {
                    Continuation::Done => {
                        return Ok(result_val);
                    }
                    Continuation::Let { name, body, env } => {
                        let mut new_env_map = env.as_ref().clone();
                        new_env_map.insert(name, result_val);
                        current_env = Rc::new(new_env_map);
                        current_term = *body;
                        continue 'trampoline; 
                    }
                    Continuation::LetRec { name, body, env } => {
                        let value_val = result_val;
                        if decode_heap_pointer(value_val).is_none() {
                            return Err(EvalError::TypeError("let rec expression must result in a heap-allocated value (a function or pair)".to_string()));
                        }
                        patch_recursively(value_val, &name, value_val, heap);
                        let mut body_env_map = env.as_ref().clone();
                        body_env_map.insert(name, value_val);
                        current_env = Rc::new(body_env_map);
                        current_term = *body;
                        continue 'trampoline; 
                    }
                    Continuation::AppFunc { arg, env } => {
                        cont_stack.push(Continuation::AppArg { func_val: result_val });
                        current_term = *arg;
                        current_env = env;
                        continue 'trampoline; 
                    }
                    Continuation::AppArg { func_val } => {
                        let arg_val = result_val;
                        if let Some(id) = decode_heap_pointer(func_val) {
                            let heap_obj = heap.get(id).cloned().ok_or(EvalError::DanglingPointerError(id))?;
                            match heap_obj {
                                HeapObject::UserFunc(closure) => {
                                                                                            let mut new_env_map = closure.env.as_ref().clone();
                                                                                            new_env_map.insert(closure.param, arg_val);
                                                                                            current_env = Rc::new(new_env_map);
                                                                                            current_term = *closure.body;
                                                                                            continue 'trampoline; 
                                                                                        }
                                HeapObject::BuiltinFunc(mut closure) => {
                                                                                            closure.args.push(arg_val);
                                                                                            if closure.args.len() == closure.arity {
                                                                                                result_val = execute_builtin(&closure.op, &closure.args, heap)?;
                                                                                                continue; // Stay in continuation loop with new value.
                                                                                            } else {
                                                                                                let new_id = heap.register(HeapObject::BuiltinFunc(closure));
                                                                                                result_val = encode_heap_pointer(new_id);
                                                                                                continue; // Stay in continuation loop with new value.
                                                                                            }
                                                                                        }
                                HeapObject::Pair(_, _) => return Err(EvalError::TypeError(format!("Cannot apply a non-function value: Pair<{}>", id))),
                                HeapObject::Tensor(_) => return Err(EvalError::TypeError(format!("Cannot apply a non-function value: Tensor<{}>", id))),
                                HeapObject::Free(_) => return Err(EvalError::DanglingPointerError(id)),
                                HeapObject::Function(_function) => return Err(EvalError::TypeError("Cannot apply a VM function in the tree-walker.".to_string())),
                                HeapObject::Closure(_) => return Err(EvalError::TypeError("Cannot apply a VM closure in the tree-walker.".to_string())),
                            }
                        } else if func_val == NIL_VALUE {
                            return Err(EvalError::TypeError("Cannot apply a non-function value: nil".to_string()));
                        } else {
                            return Err(EvalError::TypeError(format!("Cannot apply a non-function value: {}", func_val)));
                        }
                    }
                    Continuation::IfCond { then_branch, else_branch, env } => {
                        let cond_val = result_val;
                        if cond_val == 0.0 || cond_val == NIL_VALUE {
                            current_term = *else_branch;
                            current_env = env;
                            continue 'trampoline; 
                        }
                        if cond_val == 1.0 {
                            current_term = *then_branch;
                            current_env = env;
                            continue 'trampoline; 
                        }
                        cont_stack.push(Continuation::IfThen { else_branch, env: env.clone(), cond_val });
                        current_term = *then_branch;
                        current_env = env;
                        continue 'trampoline; 
                    }
                    Continuation::IfThen { else_branch, env, cond_val } => {
                        let then_val = result_val;
                        cont_stack.push(Continuation::IfElse { then_val, cond_val });
                        current_term = *else_branch;
                        current_env = env;
                        continue 'trampoline; 
                    }
                    Continuation::IfElse { then_val, cond_val } => {
                        let else_val = result_val;
                        let weight = cond_val.max(0.0).min(1.0);
                        result_val = weight * then_val + (1.0 - weight) * else_val;
                        continue; // Stay in continuation loop with new value.
                    }
                }
            }
        }
    }
}

// Calculus helpers
fn numerical_derivative(func: f64, x: f64, heap: &mut Heap) -> Result<f64, EvalError> {
    let h = 1e-6; // Small step size
    
    // f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    let f_2h = apply_function(func, x + 2.0*h, heap)?;
    let f_h = apply_function(func, x + h, heap)?;
    let f_minus_h = apply_function(func, x - h, heap)?;
    let f_minus_2h = apply_function(func, x - 2.0*h, heap)?;
    
    Ok((-f_2h + 8.0*f_h - 8.0*f_minus_h + f_minus_2h) / (12.0 * h))
}

fn numerical_integration(func: f64, a: f64, b: f64, heap: &mut Heap) -> Result<f64, EvalError> {
    // println!("numerical_integration: func={}, a={}, b={}", func, a, b);
    
    let n = 1000;
    let h = (b - a) / n as f64;
    let mut sum = 0.0;
    
    for i in 0..=n {
        let x = a + i as f64 * h;
        let weight = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        
        let fx = apply_function(func, x, heap)?;
        
        if fx.is_nan() {
            println!("Function returned NaN at x={}", x);
            return Ok(f64::NAN);
        }
        
        sum += weight * fx;
    }
    
    let result = sum * h / 3.0;
    // println!("Integration result: {}", result);
    Ok(result)
}

// --- Builtin Logic Helpers ---
fn get_builtin_arity(op: &str) -> Result<usize, EvalError> {
    if let Some(arity) = ml::get_ml_builtin_arity(op) {
        return Ok(arity);
    }
    if let Some(arity) = math::get_math_builtin_arity(op) {
        return Ok(arity);
    }
    match op {
        // Nullary
        "read-char" | "read-line" => Ok(0),
        // Unary
        "neg" | "abs" | "fuzzy_not" | "car" | "cdr" | "print" | "length"
        => Ok(1),
        // Binary
        "+" | "-" | "*" | "/" | "==" | "eq?" | "<" | ">" | "<=" | ">=" | "min" | "max" |
        "cons" | "fuzzy_and" | "fuzzy_or" | "rem" | "div" | "map" | "filter" | "diff"
        => Ok(2),
        // Ternary
        "foldl" | "integrate" => Ok(3),
        _ => Err(EvalError::TypeError(format!("Unknown builtin: {}", op))),
    }
}

pub fn list_to_vec(mut list_ptr: f64, heap: &Heap) -> Result<Vec<f64>, EvalError> {
    let mut vec = Vec::new();
    loop {
        if list_ptr == NIL_VALUE { break; }
        let (car, cdr) = if let Some(id) = decode_heap_pointer(list_ptr) {
            if let Some(HeapObject::Pair(car, cdr)) = heap.get(id) { (*car, *cdr) } 
            else { return Err(EvalError::TypeError("Expected a proper list.".to_string())); }
        } else { return Err(EvalError::TypeError("Expected a proper list.".to_string())); };
        vec.push(car);
        list_ptr = cdr;
    }
    Ok(vec)
}

pub fn vec_to_list(vec: &[f64], heap: &mut Heap) -> f64 {
    let mut list = NIL_VALUE;
    for &val in vec.iter().rev() {
        let new_pair = HeapObject::Pair(val, list);
        let id = heap.register(new_pair);
        list = encode_heap_pointer(id);
    }
    list
}

fn execute_builtin(op: &str, args: &[f64], heap: &mut Heap) -> Result<f64, EvalError> {
    if math::get_math_builtin_arity(op).is_some() {
        return math::execute_math_builtin(op, args);
    }
    if ml::get_ml_builtin_arity(op).is_some() {
        return ml::execute_ml_builtin(op, args, heap);
    }
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

        // Calculus
        "diff" => {
            let func = args[0];
            let point = args[1];
            numerical_derivative(func, point, heap)
        }
        "integrate" => {
            let func = args[0];
            let a = args[1];
            let b = args[2];
            numerical_integration(func, a, b, heap)
        }

        // Standard arithmetic/logic (no heap access)
        "neg" => Ok(-args[0]),
        "abs" => Ok(args[0].abs()),
        "sqrt" => Ok(if args[0] < 0.0 { f64::NAN } else { args[0].sqrt() }),
        "fuzzy_not" => Ok(1.0 - args[0].max(0.0).min(1.0)),
        "exp" => Ok(args[0].exp()), 
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
    let diff = (x - y).abs();
    // Use max of absolute values of x and y, but at least 1.0 to avoid inflating the result
    // for small numbers and to prevent division by zero.
    let scale_factor = x.abs().max(y.abs()).max(1.0);
    (-(diff / scale_factor)).exp()
}