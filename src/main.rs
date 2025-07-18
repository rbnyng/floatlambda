// src/main.rs

// FloatLambda
// A lambda calculus-like language where everything is an f64.

use clap::Parser as ClapParser;
use float_lambda::memory::encode_heap_pointer;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::rc::Rc;

// Import the necessary components from our library crate.
use float_lambda::{
    ast::Term,
    memory::{Heap, HeapObject, decode_heap_pointer, NIL_VALUE},
    parser::parse,
};

/// The prelude script, containing standard library functions written in FloatLambda.
/// This is automatically loaded when the interpreter starts by embedding the file at compile time.
const PRELUDE_SRC: &str = include_str!("prelude.fl");

#[derive(ClapParser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// The script file to run. If not provided, launches the REPL.
    file: Option<PathBuf>,
}

fn show_examples() {
    println!("\n--- FloatLambda Examples ---\n");

    let examples = [
        ("Simple arithmetic", "((+ 10) 5)"),
        ("Let-bound lambda", "let id = (λx.x) in (id 42)"),
        ("Factorial (recursive)", "let rec fac = (λn.if (< n 2) then 1 else ((* n) (fac ((- n) 1)))) in (fac 5)"),
        ("Creating a list", "(cons 1 (cons 2 (cons 3 nil)))"),
        ("Accessing list elements", "let mylist = (cons 10 (cons 20 nil)) in (car (cdr mylist))"),
        ("List of functions", "let mylist = (cons (+ 1) (cons (* 2) nil)) in ((car mylist) 99)"),
        ("Higher-order map function", "let rec map = (λf.λl.if (eq? l nil) then nil else (cons (f (car l)) (map f (cdr l)))) in let l = (cons 1 (cons 2 nil)) in (car (map (+ 10) l))"),
    ];

    for (description, code) in examples.iter() {
        println!("// {}", description);
        println!("{}\n", code);
    }
    println!("-----------------------------\n");
}

// REPL helper function to orchestrate parsing and evaluation.
fn process_input(
    input: &str,
    heap: &mut Heap,
    global_env_map: &mut HashMap<String, f64>,
    is_prelude: bool,
) -> Result<f64, String> {
    let term = parse(input).map_err(|e| format!("Parse error: {}", e))?;
    if !is_prelude {
        println!("Parsed: {}", term);
    }
    
    // If this is the prelude, we need to walk the entire let-chain and populate
    // the global environment with all definitions.
    if is_prelude {
        let mut current_term = &term;
        loop {
            match current_term {
                Term::Let(name, value, body) => {
                    let eval_env = Rc::new(global_env_map.clone());
                    let value_val = value.eval(&eval_env, heap)
                        .map_err(|e| format!("Eval error in prelude binding '{}': {}", name, e))?;
                    global_env_map.insert(name.clone(), value_val);
                    current_term = body; // Recurse into the body
                },
                Term::LetRec(name, value, body) => {
                    let eval_env = Rc::new(global_env_map.clone());
                    let get_func_term = Term::LetRec(name.clone(), value.clone(), Box::new(Term::Var(name.clone())));
                    let func_val = get_func_term.eval(&eval_env, heap)
                         .map_err(|e| format!("Eval error in prelude binding '{}': {}", name, e))?;
                    global_env_map.insert(name.clone(), func_val);
                    current_term = body; // Recurse into the body
                },
                _ => break, // Stop when we hit the final expression (e.g., 'identity')
            }
        }
    } 
    // This handles REPL-style single definitions
    else if let Term::Let(name, value, _) = &term {
        let eval_env = Rc::new(global_env_map.clone());
        let value_val = value.eval(&eval_env, heap)
            .map_err(|e| format!("Eval error: {}", e))?;
        global_env_map.insert(name.clone(), value_val);
    } else if let Term::LetRec(name, value, _) = &term {
         let eval_env = Rc::new(global_env_map.clone());
         let get_func_term = Term::LetRec(name.clone(), value.clone(), Box::new(Term::Var(name.clone())));
         let func_val = get_func_term.eval(&eval_env, heap)
              .map_err(|e| format!("Eval error: {}", e))?;
         global_env_map.insert(name.clone(), func_val);
    }
    
    // Finally, evaluate the complete term in the context of the (now correctly populated) environment.
    let final_env = Rc::new(global_env_map.clone());
    term.eval(&final_env, heap).map_err(|e| format!("Eval error: {}", e))
}

// --- HELPERS FOR HEAP INSPECTOR ---

/// Formats a single f64 value for inspection, showing if it's a number, nil, or pointer.
fn format_value(val: f64, heap: &Heap) -> String {
    if val == NIL_VALUE {
        "nil".to_string()
    } else if let Some(id) = decode_heap_pointer(val) {
        let obj_type = match heap.get(id) {
            Some(HeapObject::UserFunc(_)) => "Function",
            Some(HeapObject::BuiltinFunc(_)) => "Builtin",
            Some(HeapObject::Pair(_, _)) => "Pair",
            Some(HeapObject::Tensor(_)) => "Tensor",
            Some(HeapObject::Free(_)) => "FreeSlot",
            None => "Invalid",
        };
        format!("{}<{}>", obj_type, id)
    } else {
        val.to_string()
    }
}

/// Recursively formats a list-like structure for pretty-printing.
fn format_list_structure(mut current_ptr: f64, heap: &Heap, max_depth: usize) -> String {
    let mut parts = Vec::new();
    for _ in 0..max_depth {
        if current_ptr == NIL_VALUE {
            return format!("({})", parts.join(" "));
        }
        if let Some(id) = decode_heap_pointer(current_ptr) {
            if let Some(HeapObject::Pair(car, cdr)) = heap.get(id) {
                parts.push(format_value(*car, heap));
                current_ptr = *cdr;
            } else {
                // Not a pair, so it's a dotted list
                parts.push(".".to_string());
                parts.push(format_value(current_ptr, heap));
                return format!("({})", parts.join(" "));
            }
        } else {
            // Not a pointer, so it's a dotted list
            parts.push(".".to_string());
            parts.push(format_value(current_ptr, heap));
            return format!("({})", parts.join(" "));
        }
    }
    // Reached max depth
    parts.push("...".to_string());
    format!("({})", parts.join(" "))
}

/// Pretty-prints the details of a HeapObject for the :inspect command.
fn print_heap_object_details(obj: &HeapObject, heap: &Heap) {
    match obj {
        HeapObject::UserFunc(c) => {
            println!("  Type: User-defined Function");
            println!("  Param: {}", c.param);
            println!("  Body: {}", c.body);
            println!("  Env captures: [{}]", c.env.keys().cloned().collect::<Vec<_>>().join(", "));
        }
        HeapObject::BuiltinFunc(c) => {
            println!("  Type: Built-in Function (Partial Application)");
            println!("  Op: {}", c.op);
            println!("  Arity: {}", c.arity);
            println!("  Args applied: {} / {}", c.args.len(), c.arity);
            let formatted_args: Vec<String> = c.args.iter().map(|&arg| format_value(arg, heap)).collect();
            println!("  Captured args: [{}]", formatted_args.join(", "));
        }
        HeapObject::Pair(car, cdr) => {
            println!("  Type: Pair (Cons Cell)");
            println!("  car: {}", format_value(*car, heap));
            println!("  cdr: {}", format_value(*cdr, heap));
            // Attempt to print as a list
            let list_repr = format_list_structure(encode_heap_pointer(heap.find_id(obj).unwrap()), heap, 10);
            println!("  List repr: {}", list_repr);
        }
        HeapObject::Tensor(t) => {
            println!("  Type: Tensor");
            println!("  Shape: {:?}", t.shape);
            let data_preview: Vec<f64> = t.data.iter().take(5).copied().collect();
            println!("  Data (preview): {:?} {}", data_preview, if t.data.len() > 5 { "..." } else { "" });
        }
        HeapObject::Free(next) => {
            println!("  Type: Free Slot");
            println!("  Points to next free slot: {:?}", if *next == u64::MAX { "None".to_string() } else { next.to_string() });
        }
    }
}

// Simple REPL
pub fn repl() {
    println!("FloatLambda REPL");
    println!("Enter expressions, 'quit', ':examples', or ':inspect <id>'");

    let mut heap = Heap::new();
    let mut global_env_map = HashMap::new();
    let mut last_result = 0.0;

    loop {
        // --- GC ---
        let mut roots: Vec<f64> = global_env_map.values().copied().collect();
        roots.push(last_result);
        heap.collect(&roots);

        print!("> ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input_str = input.trim();

        if input_str == "quit" || input_str == "exit" { break; }
        if input_str.is_empty() { continue; }
        if input_str == ":examples" {
            show_examples();
            continue;
        }
        if input_str.starts_with(":inspect") {
            let parts: Vec<&str> = input_str.split_whitespace().collect();
            if parts.len() == 2 {
                match parts[1].parse::<u64>() {
                    Ok(id) => match heap.get(id) {
                        Some(obj) => {
                            println!("Heap Object [{}]:", id);
                            print_heap_object_details(obj, &heap);
                        }
                        None => println!("Error: No object found at heap ID {}.", id),
                    },
                    Err(_) => println!("Error: Invalid heap ID '{}'. Must be a number.", parts[1]),
                }
            } else {
                println!("Usage: :inspect <heap_id>");
            }
            continue;
        }

        match process_input(input_str, &mut heap, &mut global_env_map, false) {
            Ok(result) => {
                last_result = result;
                print_result(result, &heap);
            }
            Err(e) => println!("Error: {}", e),
        }
    }
}

fn main() {
    let cli = Cli::parse();

    // --- SETUP: Create heap and env, then load prelude ---
    let mut heap = Heap::new();
    let mut global_env_map = HashMap::new();

    if let Err(e) = process_input(PRELUDE_SRC, &mut heap, &mut global_env_map, true) {
        eprintln!("Fatal Error loading prelude: {}", e);
        std::process::exit(1);
    }

    if let Some(path) = cli.file {
        // A file path was provided, so we run the script.
        if let Err(e) = run_script(&path, &mut heap, &mut global_env_map) { 
            // If an error occurs, print it to stderr and exit with a non-zero code.
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    } else {
        // No file path was provided, so we launch the interactive REPL.
        repl();
    }
}

/// Runs the interpreter on a given script file.
fn run_script(path: &Path, heap: &mut Heap, global_env_map: &mut HashMap<String, f64>) -> Result<(), String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file '{}': {}", path.display(), e))?;

    // The existing process_input function can be reused here.
    let result = process_input(&content, heap, global_env_map, false)?;

    // Print the final result of the script.
    print_result(result, heap);
    Ok(())
}

/// Helper function to print a result value, checking if it's a function.
/// This avoids duplicating logic between the REPL and the script runner.
fn print_result(result: f64, heap: &Heap) {
    if let Some(id) = decode_heap_pointer(result) {
        let obj_type = match heap.get(id) {
            Some(HeapObject::UserFunc(_)) => "Function",
            Some(HeapObject::BuiltinFunc(_)) => "Builtin",
            Some(HeapObject::Pair(_, _)) => "Pair",
            Some(HeapObject::Tensor(_)) => "Tensor",
            Some(HeapObject::Free(_)) => "FreeSlot",
            None => "Invalid",
        };
        println!("Result: {}<{}> ({} objects alive)", obj_type, id, heap.alive_count());
    } else if result == NIL_VALUE {
        println!("Result: nil ({} objects alive)", heap.alive_count());
    }
    else {
        println!("Result: {} ({} objects alive)", result, heap.alive_count());
    }
}