// FloatLambda
// A lambda calculus-like language where everything is an f64.

use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use clap::Parser as ClapParser;
use std::path::{Path, PathBuf};
use rand::Rng;

#[derive(ClapParser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// The script file to run. If not provided, launches the REPL.
    file: Option<PathBuf>,
}

// --- Core Data Structures ---
pub const NIL_VALUE: f64 = f64::NEG_INFINITY;

#[derive(Debug, Clone)]
pub struct Closure {
    param: String,
    body: Box<Term>,
    env: Environment,
}

#[derive(Debug, Clone)]
pub struct BuiltinClosure {
    op: String,
    arity: usize,
    args: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum HeapObject {
    UserFunc(Closure),
    BuiltinFunc(BuiltinClosure),
    Pair(f64, f64), // The classic "cons cell" for building lists.
    Free(u64), // Points to the next free slot
}

// A central table to store all living heap-allocated objects.
pub struct Heap {
    objects: Vec<Option<HeapObject>>,
    free_list_head: Option<u64>,
}

impl Heap {
    pub fn get_mut(&mut self, id: u64) -> Option<&mut HeapObject> {
        self.objects.get_mut(id as usize).and_then(|f| f.as_mut())
    }

    pub fn new() -> Self {
        Self { objects: Vec::new(), free_list_head: None }
    }

    pub fn register(&mut self, obj: HeapObject) -> u64 {
        if let Some(free_index) = self.free_list_head {
            // Pop the head of the free list
            let next_free = self.objects[free_index as usize].take();
            self.free_list_head = if let Some(HeapObject::Free(next)) = next_free {
                Some(next)
            } else {
                None // Should not happen in a correct implementation
            };
            
            // Place the new object in the reclaimed slot
            self.objects[free_index as usize] = Some(obj);
            return free_index;
        }

        // If the free list is empty, fall back to the old method
        self.objects.push(Some(obj));
        (self.objects.len() - 1) as u64
    }

    pub fn get(&self, id: u64) -> Option<&HeapObject> {
        self.objects.get(id as usize).and_then(|f| f.as_ref())
    }

    // The garbage collector
    pub fn collect(&mut self, roots: &[f64]) {
        let mut marked = vec![false; self.objects.len()];
        let mut worklist: Vec<u64> = roots
            .iter()
            .filter_map(|val| decode_heap_pointer(*val))
            .collect();
        
        while let Some(id) = worklist.pop() {
            if id as usize >= marked.len() || marked[id as usize] {
                continue;
            }
            marked[id as usize] = true;

            // The GC must trace through all heap object types.
            if let Some(obj) = self.get(id) {
                match obj {
                    HeapObject::UserFunc(closure) => {
                        // A closure is a root for objects in its environment.
                        for val in closure.env.values() {
                            if let Some(child_id) = decode_heap_pointer(*val) {
                                worklist.push(child_id);
                            }
                        }
                    }
                    HeapObject::Pair(car, cdr) => {
                        // A pair is a root for the objects in its car and cdr.
                        if let Some(car_id) = decode_heap_pointer(*car) {
                            worklist.push(car_id);
                        }
                        if let Some(cdr_id) = decode_heap_pointer(*cdr) {
                            worklist.push(cdr_id);
                        }
                    }
                    HeapObject::BuiltinFunc(_) => {
                        // Builtins don't hold references to other heap objects.
                    }
                    HeapObject::Free(_) => {
                        
                    }
                }
            }
        }

        // --- Sweep Phase ---
        self.free_list_head = None; // Reset the free list
        for i in (0..self.objects.len()).rev() { // Iterate backwards
            if !marked[i] {
                // This object is garbage, add it to the free list.
                let next_free = self.free_list_head.unwrap_or(u64::MAX); // Use a sentinel
                self.objects[i] = Some(HeapObject::Free(next_free));
                self.free_list_head = Some(i as u64);
            }
        }
    }

    // Helper for debugging in the REPL
    pub fn alive_count(&self) -> usize {
        self.objects
            .iter()
            .filter(|o| match o {
                // Free slots and empty slots are not "alive"
                Some(HeapObject::Free(_)) | None => false,
                // Any other Some(...) variant is alive
                Some(_) => true,
            })
            .count()
    }
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

// --- NaN-Boxing Crazy Town ---

const QNAN: u64 = 0x7ff8000000000000;
const PAYLOAD_MASK: u64 = 0x0000ffffffffffff;

fn encode_heap_pointer(id: u64) -> f64 {
    f64::from_bits(QNAN | id)
}

fn decode_heap_pointer(val: f64) -> Option<u64> {
    if (val.to_bits() & 0x7ff8000000000000) == QNAN {
        Some(val.to_bits() & PAYLOAD_MASK)
    } else {
        None
    }
}

// AST Definition
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    Float(f64),
    Var(String),
    Lam(String, Box<Term>),
    App(Box<Term>, Box<Term>),
    Builtin(String),
    If(Box<Term>, Box<Term>, Box<Term>),
    Let(String, Box<Term>, Box<Term>),
    LetRec(String, Box<Term>, Box<Term>),
    Nil,
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Float(n) => write!(f, "{}", n),
            Term::Var(name) => write!(f, "{}", name),
            Term::Lam(param, body) => write!(f, "λ{}.{}", param, body),
            Term::App(func, arg) => write!(f, "({} {})", func, arg),
            Term::Builtin(name) => write!(f, "{}", name),
            Term::If(cond, then, else_) => write!(f, "if {} then {} else {}", cond, then, else_),
            Term::Let(name, val, body) => write!(f, "let {} = {} in {}", name, val, body),
            Term::LetRec(name, val, body) => write!(f, "let rec {} = {} in {}", name, val, body),
            Term::Nil => write!(f, "nil"),
        }
    }
}

type Environment = Rc<HashMap<String, f64>>;

#[derive(Debug, PartialEq)]
pub enum EvalError {
    UnboundVariable(String),
    TypeError(String),
    ArithmeticError(String),
    ParseError(String),
    DanglingPointerError(u64),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::UnboundVariable(var) => write!(f, "Unbound variable: '{}'", var),
            EvalError::TypeError(msg) => write!(f, "Type error: {}", msg),
            EvalError::ArithmeticError(msg) => write!(f, "Arithmetic error: {}", msg),
            EvalError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            EvalError::DanglingPointerError(id) => write!(f, "Dangling heap pointer: ID {} is invalid.", id),
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
                
                // Evaluate the expression to get the value (e.g., a function or a pair containing functions)
                let value_val = value.eval(&temp_env, heap)?;

                if decode_heap_pointer(value_val).is_none() {
                    return Err(EvalError::TypeError(
                        "let rec expression must result in a heap-allocated value (a function or pair)".to_string()
                    ));
                }

                // This helper function walks the created value and patches any closures
                // that captured the placeholder 'name'.
                fn patch_recursively(val: f64, name: &str, final_val: f64, heap: &mut Heap) {
                    // A variable to hold the pair's values outside the borrow scope.
                    let mut pair_to_trace: Option<(f64, f64)> = None;
                
                    if let Some(id) = decode_heap_pointer(val) {
                        if let Some(true) = heap.objects.get(id as usize).map(|o| o.is_some()) {
                            if let Some(obj) = heap.get_mut(id) { // Mutable borrow starts here.
                                match obj {
                                    HeapObject::UserFunc(closure) => {
                                        // This logic is fine as it doesn't recurse.
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
                                        // INSTEAD of recursing, save the values.
                                        pair_to_trace = Some((*car, *cdr));
                                    },
                                    _ => {}
                                }
                            } // Mutable borrow ends here.
                        }
                    }
                
                    // Now that the borrow is released, we can safely make the recursive calls.
                    if let Some((car, cdr)) = pair_to_trace {
                        patch_recursively(car, name, final_val, heap);
                        patch_recursively(cdr, name, final_val, heap);
                    }
                }

                // Start the patching process on the newly created value.
                patch_recursively(value_val, &name, value_val, heap);
            
                // Finally, evaluate the body in an environment where 'name' is bound.
                let mut body_env_map = env.as_ref().clone();
                body_env_map.insert(name.clone(), value_val);
                body.eval(&Rc::new(body_env_map), heap)
            }

            Term::Builtin(op) => {
                let arity = get_builtin_arity(op)?;
                // If arity is 0, execute immediately. Otherwise, create a curried function.
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

                if let Some(id) = decode_heap_pointer(func_val) {
                    let arg_val = arg.eval(env, heap)?;
                    
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
            
            Term::If(cond, then_branch, else_branch) => {
                let condition_val = cond.eval(env, heap)?;

                // Fuzzy if is compatible with boolean-like values.
                // A strict 0.0 or the nil value results in the else branch.
                // A strict 1.0 results in the then branch.
                // Everything else is blended like expected values.
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
        "neg" | "abs" | "sqrt" | "fuzzy_not" | "car" | "cdr" | "print" => Ok(1),
        // Binary
        "+" | "-" | "*" | "/" | "==" | "eq?" | "<" | ">" | "<=" | ">=" | "min" | "max" | "cons" | "fuzzy_and" | "fuzzy_or" | "rem" | "div" => Ok(2),
        _ => Err(EvalError::TypeError(format!("Unknown builtin: {}", op))),
    }
}

fn execute_builtin(op: &str, args: &[f64], heap: &mut Heap) -> Result<f64, EvalError> {
    match op {
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
            let p1 = decode_heap_pointer(args[0]);
            let p2 = decode_heap_pointer(args[1]);
            if p1.is_some() || p2.is_some() {
                // If either is a pointer, do pointer equality.
                Ok(if p1 == p2 { 1.0 } else { 0.0 })
            } else if args[0] == NIL_VALUE || args[1] == NIL_VALUE {
                 // If either is nil, do direct equality.
                Ok(if args[0] == args[1] { 1.0 } else { 0.0 })
            }
            else {
                // Otherwise, do fuzzy numeric equality.
                Ok(fuzzy_eq(args[0], args[1]))
            }
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

// --- The Parser ---
#[derive(Debug, PartialEq)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, PartialEq)]
pub enum ParseErrorKind {
    UnexpectedChar(char),
    UnexpectedEnd,
    InvalidNumber(String),
    InvalidSyntax(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parse error at {}:{}: {}", self.line, self.col, self.kind)
    }
}

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseErrorKind::UnexpectedChar(c) => write!(f, "Unexpected character: '{}'", c),
            ParseErrorKind::UnexpectedEnd => write!(f, "Unexpected end of input"),
            ParseErrorKind::InvalidNumber(s) => write!(f, "Invalid number: '{}'", s),
            ParseErrorKind::InvalidSyntax(s) => write!(f, "Invalid syntax: {}", s),
        }
    }
}

pub struct Parser {
    input: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Parser {
    pub fn new(input: &str) -> Self {
        Parser {
            input: input.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn current_char(&self) -> Option<char> { self.input.get(self.pos).copied() }
    fn peek_char(&self) -> Option<char> { self.input.get(self.pos + 1).copied() }

    fn advance(&mut self) {
        if let Some(c) = self.current_char() {
            if c == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            self.pos += 1;
        }
    }

    fn skip_whitespace(&mut self) {
        loop {
            // First, skip all standard whitespace characters.
            while let Some(c) = self.current_char() {
                if c.is_whitespace() {
                    self.advance();
                } else {
                    break;
                }
            }

            // After skipping whitespace, check if we're at a comment.
            if self.current_char() == Some('#') {
                // If so, consume characters until a newline or end of input.
                while let Some(c) = self.current_char() {
                    if c == '\n' {
                        break;
                    }
                    self.advance();
                }
                // The loop continues, so we'll skip the newline on the next iteration.
            } else {
                // If we're not at a comment, we're done.
                break;
            }
        }
    }

    fn error(&self, kind: ParseErrorKind) -> ParseError {
        ParseError { kind, line: self.line, col: self.col }
    }

    pub fn parse(&mut self) -> Result<Term, ParseError> {
        self.skip_whitespace();
        let term = self.parse_term()?;
        self.skip_whitespace();
        if let Some(c) = self.current_char() {
            Err(self.error(ParseErrorKind::UnexpectedChar(c)))
        } else {
            Ok(term)
        }
    }
    
    fn parse_term(&mut self) -> Result<Term, ParseError> {
        self.skip_whitespace();
        match self.current_char() {
            Some('(') => self.parse_application(),
            Some('λ') | Some('\\') => self.parse_lambda(),
            Some(c) if c.is_digit(10) || (c == '-' && self.peek_char().map_or(false, |c| c.is_digit(10))) => self.parse_number(),
            Some(c) if c.is_alphabetic() => {
                let word = self.peek_exact_word();
                match word.as_str() {
                    "if" => self.parse_if(),
                    "let" => self.parse_let(),
                    "nil" => {
                        self.consume_keyword("nil")?;
                        Ok(Term::Nil)
                    }
                    _ => self.parse_identifier(),
                }
            }
            Some(_) => self.parse_identifier(),
            None => Err(self.error(ParseErrorKind::UnexpectedEnd)),
        }
    }

    fn peek_exact_word(&self) -> String {
        let mut word = String::new();
        let mut temp_pos = self.pos;
        while let Some(&c) = self.input.get(temp_pos) {
            if c.is_alphabetic() { // Keywords are alphabetic only
                word.push(c);
                temp_pos += 1;
            } else {
                break;
            }
        }
        word
    }

    fn consume_keyword(&mut self, keyword: &str) -> Result<(), ParseError> {
        self.skip_whitespace();
        let start_line = self.line;
        let start_col = self.col;
        if self.input.get(self.pos..self.pos + keyword.len()).map_or(false, |s| s.iter().collect::<String>() == keyword) {
            // Check for word boundary
            if self.input.get(self.pos + keyword.len()).map_or(true, |c| !c.is_alphanumeric()) {
                for _ in 0..keyword.len() {
                    self.advance();
                }
                return Ok(());
            }
        }
        Err(ParseError {
            kind: ParseErrorKind::InvalidSyntax(format!("Expected keyword '{}'", keyword)),
            line: start_line,
            col: start_col,
        })
    }

    fn parse_let(&mut self) -> Result<Term, ParseError> {
        self.consume_keyword("let")?;
        self.skip_whitespace();

        // Check for the 'rec' keyword.
        let is_rec = if self.peek_exact_word() == "rec" {
            self.consume_keyword("rec")?;
            true
        } else {
            false
        };

        self.skip_whitespace();
        let name = self.parse_identifier_string()?;
        self.skip_whitespace();
        self.consume_keyword("=")?;
        let value = self.parse_term()?;
        self.skip_whitespace();
        self.consume_keyword("in")?;
        let body = self.parse_term()?;

        if is_rec {
            Ok(Term::LetRec(name, Box::new(value), Box::new(body)))
        } else {
            Ok(Term::Let(name, Box::new(value), Box::new(body)))
        }
    }

    fn parse_if(&mut self) -> Result<Term, ParseError> {
        self.consume_keyword("if")?;
        let cond = self.parse_term()?;
        self.consume_keyword("then")?;
        let then_branch = self.parse_term()?;
        self.consume_keyword("else")?;
        let else_branch = self.parse_term()?;
        Ok(Term::If(Box::new(cond), Box::new(then_branch), Box::new(else_branch)))
    }

    fn parse_application(&mut self) -> Result<Term, ParseError> {
        let start_line = self.line;
        let start_col = self.col;

        self.advance(); // consume '('
        let mut terms = Vec::new();
        loop {
            self.skip_whitespace();
            if self.current_char() == Some(')') {
                self.advance();
                break;
            }
            if self.current_char().is_none() {
                return Err(self.error(ParseErrorKind::UnexpectedEnd));
            }
            terms.push(self.parse_term()?);
        }
        if terms.is_empty() {
            return Err(ParseError {
                kind: ParseErrorKind::InvalidSyntax("Empty application () is not allowed".to_string()),
                line: start_line,
                col: start_col,
            });
        }
        let mut app = terms.remove(0);
        for term in terms {
            app = Term::App(Box::new(app), Box::new(term));
        }
        Ok(app)
    }

    fn parse_lambda(&mut self) -> Result<Term, ParseError> {
        self.advance(); // consume 'λ' or '\'
        self.skip_whitespace();
        let param = self.parse_identifier_string()?;
        self.skip_whitespace();
        if self.current_char() != Some('.') {
            return Err(self.error(ParseErrorKind::InvalidSyntax("Expected '.' in lambda".to_string())));
        }
        self.advance(); // consume '.'
        let body = self.parse_term()?;
        Ok(Term::Lam(param, Box::new(body)))
    }

    fn parse_number(&mut self) -> Result<Term, ParseError> {
        let start_line = self.line;
        let start_col = self.col;
        let mut s = String::new();

        // Handle negative numbers
        if self.current_char() == Some('-') {
            s.push('-');
            self.advance();
        }

        // Digits before decimal
        while let Some(c) = self.current_char() {
            if c.is_digit(10) {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Decimal part
        if self.current_char() == Some('.') {
            s.push('.');
            self.advance();
            while let Some(c) = self.current_char() {
                if c.is_digit(10) {
                    s.push(c);
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // Scientific notation (e or E)
        if let Some(c) = self.current_char() {
            if c == 'e' || c == 'E' {
                s.push(c);
                self.advance();
                // Optional sign for exponent
                if let Some(sign_c) = self.current_char() {
                    if sign_c == '+' || sign_c == '-' {
                        s.push(sign_c);
                        self.advance();
                    }
                }
                // Exponent digits
                while let Some(c_exp) = self.current_char() {
                    if c_exp.is_digit(10) {
                        s.push(c_exp);
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }

        s.parse::<f64>()
            .map(Term::Float)
            .map_err(|_| ParseError {
                kind: ParseErrorKind::InvalidNumber(s),
                line: start_line,
                col: start_col,
            })
    }

    fn parse_identifier(&mut self) -> Result<Term, ParseError> {
        let name = self.parse_identifier_string()?;
        match name.as_str() {
            "neg" | "abs" | "sqrt" | "fuzzy_not" | "+" | "-" | "*" | "/" | "==" | "eq?" |
            "<" | ">" | "<=" | ">=" | "min" | "max" | "cons" | "car" | "cdr" |
            "fuzzy_and" | "fuzzy_or" | "rem" | "div" |
            "print" | "read-char" | "read-line" => Ok(Term::Builtin(name)), // Add new builtins
            _ => Ok(Term::Var(name)),
        }
    }

    fn parse_identifier_string(&mut self) -> Result<String, ParseError> {
        let start_line = self.line;
        let start_col = self.col;
        let mut name = String::new();

        if let Some(c) = self.current_char() {
            if "+-*/=<>!".contains(c) {
                name.push(c);
                self.advance();
                if let Some(c2) = self.current_char() {
                    if "=<>".contains(c2) {
                        name.push(c2);
                        self.advance();
                    }
                }
                return Ok(name);
            }
        }
        while let Some(c) = self.current_char() {
            if c.is_alphanumeric() || c == '_' || c == '?' {
                name.push(c);
                self.advance();
            } else {
                break;
            }
        }
        if name.is_empty() {
            Err(ParseError {
                kind: ParseErrorKind::InvalidSyntax("Expected an identifier".to_string()),
                line: start_line,
                col: start_col,
            })
        } else {
            Ok(name)
        }
    }
}

// Convenience function for parsing
pub fn parse(input: &str) -> Result<Term, ParseError> {
    Parser::new(input).parse()
}

// REPL helper function to orchestrate parsing and evaluation.
fn process_input(
    input: &str,
    heap: &mut Heap,
    global_env_map: &mut HashMap<String, f64>,
) -> Result<f64, String> {
    let term = parse(input).map_err(|e| format!("Parse error: {}", e))?;
    println!("Parsed: {}", term);

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

        match process_input(input_str, &mut heap, &mut global_env_map) {
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
    let result = process_input(&content, &mut table, &mut global_env_map)?;

    // Print the final result of the script.
    print_result(result, &table);
    Ok(())
}

/// Helper function to print a result value, checking if it's a function.
/// This avoids duplicating logic between the REPL and the script runner.
fn print_result(result: f64, heap: &Heap) {
    if let Some(id) = decode_heap_pointer(result) {
        // --- CHANGE: Provide more detail about the heap object.
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

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to run evaluation
    fn eval_ok(input: &str) -> f64 {
        let term = parse(input).unwrap();
        let mut heap = Heap::new();
        term.eval(&Rc::new(HashMap::new()), &mut heap).unwrap()
    }

    fn eval_err(input: &str) -> EvalError {
        let term = parse(input).unwrap();
        let mut heap = Heap::new();
        term.eval(&Rc::new(HashMap::new()), &mut heap).unwrap_err()
    }

    #[test]
    fn test_float_eval() {
        assert_eq!(eval_ok("42.0"), 42.0);
    }

    #[test]
    fn test_var_eval() {
        // Need to use let to bind variables now, as direct env manipulation is harder in tests
        assert_eq!(eval_ok("let x = 10.0 in x"), 10.0);
        // Test unbound variable error
        assert_eq!(
            eval_err("x"),
            EvalError::UnboundVariable("x".to_string())
        );
    }

    #[test]
    fn test_fuzzy_eq() {
        assert!((fuzzy_eq(1.0, 1.0) - 1.0).abs() < 0.001);
        assert!(fuzzy_eq(1.0, 2.0) < 1.0);
        assert!(fuzzy_eq(1000.0, 1001.0) < 0.5);
    }

    #[test]
    fn test_lambda_application() {
        // (λx.x) 42.0
        assert_eq!(eval_ok("(λx.x 42.0)"), 42.0);
    }

    #[test]
    fn test_binary_operation() {
        // ((+ 1.0) 2.0) should equal 3.0
        assert_eq!(eval_ok("((+ 1.0) 2.0)"), 3.0);
        assert_eq!(eval_ok("((/ 10.0) 2.0)"), 5.0);
        assert_eq!(eval_ok("((/ 10.0) 0.0)"), f64::INFINITY);
    }

    #[test]
    fn test_parse_float() {
        assert_eq!(parse("42.0").unwrap(), Term::Float(42.0));
        assert_eq!(parse("-3.14").unwrap(), Term::Float(-3.14));
        assert_eq!(parse("0.0").unwrap(), Term::Float(0.0));
    }

    #[test]
    fn test_parse_variable() {
        assert_eq!(parse("x").unwrap(), Term::Var("x".to_string()));
        assert_eq!(parse("foo_bar").unwrap(), Term::Var("foo_bar".to_string()));
    }

    #[test]
    fn test_parse_builtin() {
        assert_eq!(parse("+").unwrap(), Term::Builtin("+".to_string()));
        assert_eq!(parse("sqrt").unwrap(), Term::Builtin("sqrt".to_string()));
    }

    #[test]
    fn test_parse_lambda() {
        let lambda = parse("λx.x").unwrap();
        assert_eq!(lambda, Term::Lam("x".to_string(), Box::new(Term::Var("x".to_string()))));
        
        // Test backslash syntax too
        let lambda2 = parse("\\x.x").unwrap();
        assert_eq!(lambda2, lambda);
    }

    #[test]
    fn test_parse_application() {
        let app = parse("(f x)").unwrap();
        assert_eq!(app, Term::App(
            Box::new(Term::Var("f".to_string())),
            Box::new(Term::Var("x".to_string()))
        ));
        
        // Test multiple arguments (parsed as curried applications)
        let app2 = parse("(+ 1.0 2.0)").unwrap();
        assert_eq!(app2, Term::App(
            Box::new(Term::App(
                Box::new(Term::Builtin("+".to_string())),
                Box::new(Term::Float(1.0))
            )),
            Box::new(Term::Float(2.0))
        ));
    }

    #[test]
    fn test_fuzzy_logic() {
        assert_eq!(eval_ok("((fuzzy_and 0.8) 0.6)"), 0.48);
        assert!((eval_ok("((fuzzy_or 0.8) 0.6)") - 0.92).abs() < 0.0001);
        assert_eq!(eval_ok("(fuzzy_not 0.3)"), 0.7);
        assert_eq!(eval_ok("(fuzzy_not 1.5)"), 0.0);
    }

    #[test]
    fn test_blended_conditional() {
        // if 0.7 then 100.0 else 200.0
        // = 0.7 * 100.0 + 0.3 * 200.0 = 70.0 + 60.0 = 130.0
        assert_eq!(eval_ok("if 0.7 then 100.0 else 200.0"), 130.0);
    }

    #[test]
    fn test_parse_if() {
        let if_expr = parse("if 0.5 then 10.0 else 20.0").unwrap();
        assert!(matches!(if_expr, Term::If(_, _, _)));
        assert_eq!(eval_ok("if 0.5 then 10.0 else 20.0"), 15.0); // 0.5 * 10.0 + 0.5 * 20.0
    }

    #[test]
    fn test_truth_decay_concept() {
        // Test that repeated operations can change "true" values
        let mut truth = 1.0;
        for _ in 0..1000 {
            truth = truth * 1.0000001;
        }
        assert!(truth != 1.0);
        assert!(truth > 1.0);
    }
    
    #[test]
    fn test_complex_fuzzy_expression() {
        // Test: if ((fuzzy_and 0.8) 0.6) then 100.0 else 200.0
        // 0.8 * 0.6 = 0.48
        // 0.48 * 100.0 + 0.52 * 200.0 = 48.0 + 104.0 = 152.0
        assert_eq!(eval_ok("if ((fuzzy_and 0.8) 0.6) then 100.0 else 200.0"), 152.0);
    }
    
    #[test]
    fn test_nan_handling() {
        // sqrt(-1) should be NaN
        let result = eval_ok("(sqrt -1.0)");
        assert!(result.is_nan());
    }
    
    #[test]
    fn test_improved_fuzzy_eq() {
        assert_eq!(fuzzy_eq(1.0, 1.0), 1.0);
        assert!((fuzzy_eq(1.0, 2.0) - 0.36787944117144233).abs() < 0.0001);
        assert!((fuzzy_eq(1000.0, 1001.0) - 0.36787944117144233).abs() < 0.0001); // e^(-1) for diff of 1
    }
    
    #[test]
    fn test_robust_parser_keywords() {
        // Test that "iff" is parsed as a variable (not the "if" keyword)
        assert!(matches!(parse("iff").unwrap(), Term::Var(_)));
        
        // Test that malformed if statements are caught
        assert!(parse("if 1 thenn 2 else 3").is_err());
        assert!(parse("if 1 then 2 elsee 3").is_err());
        
        // Test that we can distinguish if from iff
        assert!(matches!(parse("if 1 then 2 else 3").unwrap(), Term::If(_, _, _)));
        
        // Test that we can use "iff" as a variable name in a lambda
        let iff_usage = parse("λiff.iff").unwrap();
        assert!(matches!(iff_usage, Term::Lam(_, _)));
        assert_eq!(eval_ok("((λiff.iff) 42.0)"), 42.0); // And it works!
    }

    // --- Tests for First-Class Functions, Currying, Let ---

    #[test]
    fn test_lone_lambda_is_a_value() {
        let result = eval_ok("λx.x");
        assert!(result.is_nan()); // It's a NaN-boxed value
        assert!(decode_heap_pointer(result).is_some()); // It decodes to a valid ID
    }
    
    #[test]
    fn test_let_binding_simple() {
        assert_eq!(eval_ok("let x = 10 in ((+ x) 5)"), 15.0);
        assert_eq!(eval_ok("let x = 3 in let y = 4 in ((* x) y)"), 12.0);
    }
    
    #[test]
    fn test_currying_builtin() {
        let code = "let add5 = (+ 5) in (add5 10)";
        assert_eq!(eval_ok(code), 15.0);
        let code2 = "let sub_from_10 = (- 10) in (sub_from_10 3)"; // 10 - 3 = 7
        assert_eq!(eval_ok(code2), 7.0);
    }
    
    #[test]
    fn test_higher_order_function_const() {
        let code = "let const = (λx.λy.x) in ((const 42) 100)";
        assert_eq!(eval_ok(code), 42.0);
    }
    
    #[test]
    fn test_higher_order_function_apply() {
        let code = "let apply = (λf.λx.(f x)) in (let add1 = (+ 1) in ((apply add1) 10))";
        assert_eq!(eval_ok(code), 11.0);
    }
    
    #[test]
    fn test_application_of_non_function() {
        assert_eq!(
            eval_err("(10 20)"),
            EvalError::TypeError("Cannot apply a non-function value: 10".to_string())
        );
    }

    #[test]
    fn test_emergent_property_nan_propagation() {
        // This test demonstrates that simple arithmetic on a function might not "break" it
        // because NaN + number often just returns the same NaN.
        let code = "let f = (λx.x) in (let g = (+ f 0) in (g 42))";
        // g is effectively the same function as f. The application should succeed.
        assert_eq!(eval_ok(code), 42.0);

        // Even NaN * 1.0 typically returns the same NaN payload.
        let code2 = "let f = (λx.x) in (let g = ((* f) 1.0) in (g 42))";
        assert_eq!(eval_ok(code2), 42.0);
    }
    
    #[test]
    fn test_if_statement_with_function_as_condition() {
        // The if statement will treat the NaN-boxed function as a number.
        // A NaN value, when clamped by max(0.0).min(1.0), typically becomes 0.0.
        // Therefore, the else branch should be taken.
        let code = "if (λx.x) then 100 else 200";
        assert_eq!(eval_ok(code), 200.0);
    }
    
    #[test]
    fn test_complex_expression_with_let() {
        let code = "let a = 10 in let b = 20 in if ((== a) 10.1) then b else a";
        let result = eval_ok(code);
        // fuzzy_eq(10.0, 10.1) is approx 0.904837
        // 0.904837 * 20.0 + (1.0 - 0.904837) * 10.0 = 18.09674 + 0.95163 = 19.04837
        assert!((result - 19.04837).abs() < 0.0001);
    }

    // --- Complex Data Structures ---

    #[test]
    fn test_list_of_lists() {
        let code = "let list = (cons (cons 1 2) (cons (cons 3 4) nil)) in (car (car (cdr list)))";
        // list = ((1 . 2) . ((3 . 4) . nil))
        // (cdr list) -> ((3 . 4) . nil)
        // (car (cdr list)) -> (3 . 4)
        // (car (car (cdr list))) -> 3
        assert_eq!(eval_ok(code), 3.0);
    }

    #[test]
    fn test_non_nil_terminated_list() {
        // A "dotted pair" at the end.
        let code = "let pair = (cons 1 2) in (cdr pair)";
        assert_eq!(eval_ok(code), 2.0);
    }

    #[test]
    fn test_can_cons_anything() {
        // (cons (λx.x) (cons 10 nil))
        let code = "let f = (λx.x) in let l = (cons f nil) in ((car l) 42)";
        assert_eq!(eval_ok(code), 42.0);
    }
    
    // --- Higher-Order Functions ---

    #[test]
    fn test_higher_order_map() {
        let map_code = "let rec map = (λf.λl.if (eq? l nil) then nil else (cons (f (car l)) (map f (cdr l))))";
        let list_code = "let mylist = (cons 10 (cons 20 (cons 30 nil)))";
        let add5 = "(λx.(+ x 5))";
        
        let full_code = format!("let apply = (λf.λx.(f x)) in {} in {} in let mapped_list = ((map {}) mylist) in (car (cdr mapped_list))", map_code, list_code, add5);
        // mapped_list would be (15 . (25 . (35 . nil)))
        // (cdr mapped_list) -> (25 . (35 . nil))
        // (car (cdr mapped_list)) -> 25
        assert_eq!(eval_ok(&full_code), 25.0);
    }
    
    #[test]
    fn test_higher_order_filter() {
        let filter_code = "let rec filter = (λp.λl.
            if (eq? l nil) then nil 
            else (
                let head = (car l) in
                let tail = (cdr l) in
                if (p head) then (cons head (filter p tail)) else (filter p tail)
            ))";
        let list_code = "let mylist = (cons 1 (cons 2 (cons 3 (cons 4 nil))))";
        
        // Let's use a predicate that works with existing builtins: is_greater_than_2
        let is_gt2 = "(λx.(> x 2))";
        
        let full_code = format!("{} in {} in let filtered_list = ((filter {}) mylist) in (car (cdr filtered_list))", filter_code, list_code, is_gt2);
        // filtered_list should be (3 . (4 . nil))
        // (cdr filtered_list) -> (4 . nil)
        // (car (cdr filtered_list)) -> 4
        assert_eq!(eval_ok(&full_code), 4.0);
    }
    
    #[test]
    fn test_higher_order_fold_left_aka_reduce() {
        let fold_code = "let rec fold = (λf.λacc.λl.
            if (eq? l nil) then acc 
            else (
                fold f (f acc (car l)) (cdr l)
            ))";
        let list_code = "let mylist = (cons 1 (cons 2 (cons 3 (cons 4 nil))))";
        let sum_op = "(λa.λb.(+ a b))";
        
        // Sum of list: fold(+, 0, list)
        let full_code = format!("{} in {} in (((fold {}) 0) mylist)", fold_code, list_code, sum_op);
        // (1 + 2 + 3 + 4) = 10
        assert_eq!(eval_ok(&full_code), 10.0);
    }

    // --- Recursion and Scope ---
    #[test]
    fn test_closure_captures_correct_env() {
        let code = "let make_adder = (λx. (λy. (+ x y))) in let add5 = (make_adder 5) in (add5 10)";
        assert_eq!(eval_ok(code), 15.0);
        
        // Ensure it doesn't capture a later binding of x
        let code2 = "let make_adder = (λx. (λy. (+ x y))) in let add5 = (make_adder 5) in let x = 100 in (add5 10)";
        assert_eq!(eval_ok(code2), 15.0);
    }
    
    // --- Garbage Collector Stress Tests ---

    #[test]
    fn test_gc_on_long_list() {
        let mut heap = Heap::new();
        let mut global_env = HashMap::new();
        
        let mut list_str = "nil".to_string();
        for i in 0..100 {
            list_str = format!("(cons {} {})", i, list_str);
        }
        
        let list_val = process_input(&list_str, &mut heap, &mut global_env).unwrap();
        assert_eq!(heap.alive_count(), 300);

        heap.collect(&[list_val]);
        assert_eq!(heap.alive_count(), 100);

        heap.collect(&[]);
        assert_eq!(heap.alive_count(), 0);
    }

    #[test]
    fn test_gc_reclaims_part_of_a_structure() {
        let mut heap = Heap::new();
        let env = Rc::new(HashMap::new());

        // 1. Evaluate the list creation expression directly.
        let list_term = parse("(cons 1 (cons 2 (cons 3 nil)))").unwrap();
        let list_val = list_term.eval(&env, &mut heap).unwrap();

        // 2. Run the GC with the list head as the root. This cleans up the temporary
        heap.collect(&[list_val]);
        assert_eq!(heap.alive_count(), 3);

        // 3. Create a new environment for the next step.
        let mut next_env_map = HashMap::new();
        next_env_map.insert("list".to_string(), list_val);
        let next_env = Rc::new(next_env_map);

        // 4. Evaluate (cdr list) in the controlled environment.
        let cdr_term = parse("(cdr list)").unwrap();
        let tail_val = cdr_term.eval(&next_env, &mut heap).unwrap();

        assert_eq!(heap.alive_count(), 4); 
        
        heap.collect(&[tail_val]);
        
        // The head of the list is no longer rooted. 2 objects should remain.
        assert_eq!(heap.alive_count(), 2);
    }

    // --- Error Condition Tests ---

    #[test]
    fn test_application_error_types() {
        assert_eq!(eval_err("(1 2)"), EvalError::TypeError("Cannot apply a non-function value: 1".to_string()));
        assert_eq!(eval_err("(nil 2)"), EvalError::TypeError("Cannot apply a non-function value: nil".to_string()));
        
        let list_ptr_code = "(cons 1 2)";
        let result = eval_ok(list_ptr_code);
        let id = decode_heap_pointer(result).unwrap();
        let expected_err = EvalError::TypeError(format!("Cannot apply a non-function value: Pair<{}>", id));
        assert_eq!(eval_err(&format!("({} 3)", list_ptr_code)), expected_err);
    }
    
    #[test]
    fn test_unbound_variable_in_deeper_expression() {
        let code = "(+ 1 (z 10))"; // z is unbound
        assert_eq!(eval_err(code), EvalError::UnboundVariable("z".to_string()));
    }
    
    #[test]
    fn test_let_rec_requires_lambda() {
        let code = "let rec x = 5 in x";
        assert_eq!(eval_err(code), EvalError::TypeError("let rec expression must result in a heap-allocated value (a function or pair)".to_string()));
    }
    
    #[test]
    fn test_car_cdr_type_errors() {
        assert_eq!(eval_err("(car 1.0)"), EvalError::TypeError("'car' expects a pair, but got a number or nil.".to_string()));
        assert_eq!(eval_err("(cdr nil)"), EvalError::TypeError("'cdr' expects a pair, but got a number or nil.".to_string()));
        let code = "(car (λx.x))";
        assert_eq!(eval_err(code), EvalError::TypeError("'car' expects a pair, but got another heap object.".to_string()));
    }

    // --- String Tests ---

    #[test]
    fn test_string_as_list_manipulation() {
        // let s = "Hi" -> (cons 72.0 (cons 105.0 nil))
        let code = "
            let s = (cons 72.0 (cons 105.0 nil)) in
            let first_char = (car s) in
            # Test comments
            let second_char = (car (cdr s)) in
            second_char
        ";
        assert_eq!(eval_ok(code), 105.0);
    
        // Also check that the list is nil-terminated correctly.
        let code_nil_check = "
            let s = (cons 72.0 (cons 105.0 nil)) in
            (eq? (cdr (cdr s)) nil)
        ";
        assert_eq!(eval_ok(code_nil_check), 1.0);
    }

    #[test]
    fn test_strlen_on_string_list() {
        // Define strlen, which finds the length of a list.
        let strlen_code = "let rec strlen = (λl. if (eq? l nil) then 0 else (+ 1 (strlen (cdr l))))";
        
        // Create the string "cat" -> (cons 99 (cons 97 (cons 116 nil)))
        let str_cat = "(cons 99.0 (cons 97.0 (cons 116.0 nil)))";
    
        let full_code = format!("
            {} in 
            (strlen {})
        ", strlen_code, str_cat);
    
        // The length of "cat" is 3.
        assert_eq!(eval_ok(&full_code), 3.0);
    }

}