// src/main.rs

// FloatLambda
// A lambda calculus-like language where everything is an f64.

use clap::Parser as ClapParser;
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
    
    // Create the evaluation environment from the REPL's global state
    let eval_env = Rc::new(global_env_map.clone());
    
    // We need to handle top-level let and let rec to update the REPL state for the *next* command.
    match &term {
        Term::Let(name, value, _) => {
            // To add a let binding, we must evaluate its value part.
            let value_val = value.eval(&eval_env, heap)
                .map_err(|e| format!("Eval error: {}", e))?;
            // And then add it to our persistent global map.
            global_env_map.insert(name.clone(), value_val);
        }
        Term::LetRec(name, value, _) => {
            // For a let rec, we need to evaluate it to get the cyclic function value.
            // We create a temporary let rec that returns the function itself.
            let get_func_term = Term::LetRec(name.clone(), value.clone(), Box::new(Term::Var(name.clone())));
            let func_val = get_func_term.eval(&eval_env, heap)
                 .map_err(|e| format!("Eval error: {}", e))?;
            // And add the resulting function to the global map.
            global_env_map.insert(name.clone(), func_val);
        }
        _ => {}
    }

    // Finally, evaluate the complete term in the context of the (possibly updated) environment
    // to get the result for *this* command.
    let final_env = Rc::new(global_env_map.clone());
    term.eval(&final_env, heap).map_err(|e| format!("Eval error: {}", e))
}

// Simple REPL
pub fn repl() {
    println!("FloatLambda v3 REPL");
    println!("Enter expressions, 'quit', or ':examples'");

    let mut heap = Heap::new(); // --- CHANGE
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
        if let Err(e) = run_script(&path) {
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
fn run_script(path: &Path) -> Result<(), String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file '{}': {}", path.display(), e))?;

    let mut table = Heap::new();
    let mut global_env_map = HashMap::new();

    // The existing process_input function can be reused here.
    let result = process_input(&content, &mut table, &mut global_env_map, false)?;

    // Print the final result of the script.
    print_result(result, &table);
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