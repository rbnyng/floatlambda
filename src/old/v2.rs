// FloatLambda v2
// A lambda calculus-like language where everything is an f64.

use std::collections::HashMap;
use std::fmt;
use std::rc::Rc; 
use clap::Parser as ClapParser;
use std::path::{Path, PathBuf};

#[derive(ClapParser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// The script file to run. If not provided, launches the REPL.
    file: Option<PathBuf>,
}

// --- Core Data Structures for First-Class Functions ---

// The runtime representation of a user-defined lambda function.
// It captures its code (param, body) and its creation environment.
#[derive(Debug, Clone)]
pub struct Closure {
    param: String,
    body: Box<Term>,
    env: Environment,
}

// The runtime representation of a builtin function, which can be partially applied.
#[derive(Debug, Clone)]
pub struct BuiltinClosure {
    op: String,
    arity: usize,
    args: Vec<f64>,
}

// A function can be either a user-defined closure or a (possibly partial) builtin.
#[derive(Debug, Clone)]
pub enum Function {
    User(Closure),
    Builtin(BuiltinClosure),
}

// A central table to store all living functions. The evaluator will pass around
// IDs (indices) into this table, encoded as f64 NaN values.
pub struct FunctionTable {
    functions: Vec<Option<Function>>,
}

impl FunctionTable {
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Function> {
        self.functions.get_mut(id as usize).and_then(|f| f.as_mut())
    }

    pub fn new() -> Self {
        Self { functions: Vec::new() }
    }

    // Reuses empty slots
    pub fn register(&mut self, func: Function) -> u64 {
        // Look for an empty slot to reuse
        for (i, slot) in self.functions.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(func);
                return i as u64;
            }
        }
        // If no empty slots, push a new one
        self.functions.push(Some(func));
        (self.functions.len() - 1) as u64
    }

    pub fn get(&self, id: u64) -> Option<&Function> {
        self.functions.get(id as usize).and_then(|f| f.as_ref())
    }

    // The garbage collector
    pub fn collect(&mut self, roots: &[f64]) {
        let mut marked = vec![false; self.functions.len()];
        let mut worklist: Vec<u64> = roots
            .iter()
            .filter_map(|val| decode_function_id(*val))
            .collect();
        
        // --- Mark Phase ---
        while let Some(id) = worklist.pop() {
            if id as usize >= marked.len() || marked[id as usize] {
                continue; // Already marked or invalid ID
            }
            marked[id as usize] = true;

            if let Some(Function::User(closure)) = self.get(id) {
                // A closure is a root for other functions in its environment
                for val in closure.env.values() {
                    if let Some(child_id) = decode_function_id(*val) {
                        worklist.push(child_id);
                    }
                }
            }
        }

        // --- Sweep Phase ---
        for i in 0..self.functions.len() {
            if !marked[i] {
                // This function is unreachable, free it.
                self.functions[i] = None;
            }
        }
    }
}

// Prints a curated list of examples.
fn show_examples() {
    println!("\n--- FloatLambda v2 Examples ---\n");

    let examples = [
        ("Simple arithmetic", "((+ 10) 5)"),
        ("Using 'let' bindings", "let x = 10 in let y = 20 in ((* x) y)"),
        ("Currying a builtin", "let add5 = (+ 5) in (add5 100)"),
        ("The identity function", "((λx.x) 42)"),
        ("A higher-order 'const' function", "let const = (λx.λy.x) in ((const 123) 456)"),
        ("A higher-order 'apply' function", "let apply = (λf.λx.(f x)) in (apply sqrt 16)"),
        ("Fuzzy equality check", "((== 5) 5.1)"),
        ("Blended 'if' statement", "if 0.25 then 100 else 0"),
        ("Putting it all together", "let sub_from = (λn.(- n)) in let tenth = ((sub_from 1) 0.9) in if ((== 5) 4.9) then 100 else tenth")
    ];

    for (description, code) in examples.iter() {
        println!("// {}", description);
        println!("{}\n", code);
    }
    println!("-----------------------------\n");
}

// --- NaN-Boxing Crazy Town ---

// A "quiet NaN" bit pattern. The exponent is all 1s, and the first bit of the mantissa is 1.
const QNAN: u64 = 0x7ff8000000000000;
// A mask to extract the 48-bit payload (our function ID) from the mantissa.
const PAYLOAD_MASK: u64 = 0x0000ffffffffffff;

// Encodes a function ID into a special f64 NaN value.
fn encode_function_id(id: u64) -> f64 {
    f64::from_bits(QNAN | id)
}

// Decodes a function ID from an f64, if it's one of our special NaNs.
fn decode_function_id(val: f64) -> Option<u64> {
    // Check if the bits match our NaN pattern.
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
    Let(String, Box<Term>, Box<Term>), // name, value, body
    LetRec(String, Box<Term>, Box<Term>), 
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
        }
    }
}

// Environment for variable bindings
type Environment = Rc<HashMap<String, f64>>;

// Evaluation errors
#[derive(Debug, PartialEq)]
pub enum EvalError {
    UnboundVariable(String),
    TypeError(String),
    ArithmeticError(String),
    ParseError(String),
    DanglingFunctionError(u64),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::UnboundVariable(var) => write!(f, "Unbound variable: '{}'", var),
            EvalError::TypeError(msg) => write!(f, "Type error: {}", msg),
            EvalError::ArithmeticError(msg) => write!(f, "Arithmetic error: {}", msg),
            EvalError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            EvalError::DanglingFunctionError(id) => write!(f, "Dangling function pointer: ID {} is invalid. The function may have been corrupted by arithmetic.", id),
        }
    }
}

// --- The Evaluator ---
impl Term {
    pub fn eval(&self, env: &Environment, table: &mut FunctionTable) -> Result<f64, EvalError> {
        match self {
            Term::Float(n) => Ok(*n),

            Term::Var(name) => env
                .get(name)
                .copied()
                .ok_or_else(|| EvalError::UnboundVariable(name.clone())),

            Term::Lam(param, body) => {
                // A lambda evaluates to a FUNCTION VALUE.
                // We create a closure, register it, and return its NaN-boxed ID.
                let closure = Closure {
                    param: param.clone(),
                    body: body.clone(),
                    env: env.clone(), // Capture the current environment
                };
                let id = table.register(Function::User(closure));
                Ok(encode_function_id(id))
            }
            
            Term::Let(name, value, body) => {
                let value_val = value.eval(env, table)?;
                // To avoid mutation, we must clone the inner map to create the new env.
                let mut new_env_map = env.as_ref().clone(); 
                new_env_map.insert(name.clone(), value_val);
                body.eval(&Rc::new(new_env_map), table)
            }

            Term::LetRec(name, value, body) => {
                // The value of a let rec must be a function.
                let lam = match &**value {
                    Term::Lam(p, b) => Term::Lam(p.clone(), b.clone()),
                    _ => return Err(EvalError::TypeError(
                        "let rec expression must be a lambda".to_string()
                    )),
                };

                // Create a temporary environment for the closure that includes a placeholder for itself.
                let mut temp_env_map = env.as_ref().clone();
                temp_env_map.insert(name.clone(), f64::NAN); // Placeholder
                let temp_env = Rc::new(temp_env_map);
                
                // Evaluate the lambda to create the closure, capturing the temp env.
                let func_val = lam.eval(&temp_env, table)?;
                let id = decode_function_id(func_val).ok_or_else(|| 
                    EvalError::TypeError("let rec expression must be a function".to_string())
                )?;

                // BACKPATCH: Update the closure's captured environment to point to itself.
                if let Some(Function::User(closure)) = table.get_mut(id) {
                    if let Some(env_map) = Rc::get_mut(&mut closure.env) {
                        env_map.insert(name.clone(), func_val);
                    } else {
                        // This fallback is unlikely but safe: if the env is already shared, clone it.
                        let mut new_map = closure.env.as_ref().clone();
                        new_map.insert(name.clone(), func_val);
                        closure.env = Rc::new(new_map);
                    }
                }

                // Finally, evaluate the body in an environment where name is properly bound.
                let mut body_env_map = env.as_ref().clone();
                body_env_map.insert(name.clone(), func_val);
                body.eval(&Rc::new(body_env_map), table)
            }

            Term::Builtin(op) => {
                // A builtin also evaluates to a FUNCTION VALUE.
                // We create a builtin closure, register it, and return its ID.
                let arity = get_builtin_arity(op)?;
                let builtin_closure = BuiltinClosure {
                    op: op.clone(),
                    arity,
                    args: Vec::new(),
                };
                let id = table.register(Function::Builtin(builtin_closure));
                Ok(encode_function_id(id))
            }

            Term::App(func, arg) => {
                let func_val = func.eval(env, table)?;

                // Check if the value is one of our NaN-boxed functions.
                if let Some(id) = decode_function_id(func_val) {
                    let arg_val = arg.eval(env, table)?;

                    // Clone the function from the table to avoid borrow checker issues with recursion
                    let function = table.get(id).cloned().ok_or(EvalError::DanglingFunctionError(id))?;

                    match function {
                        Function::User(closure) => {
                            // Apply a user-defined lambda
                            let mut new_env_map = closure.env.as_ref().clone(); 
                            new_env_map.insert(closure.param, arg_val);
                            closure.body.eval(&Rc::new(new_env_map), table)
                        }

                        Function::Builtin(mut closure) => {
                            // Apply a builtin function
                            closure.args.push(arg_val);
                            if closure.args.len() == closure.arity {
                                // Arity met, execute the builtin
                                execute_builtin(&closure.op, &closure.args)
                            } else {
                                // Arity not met, return a NEW partially applied function
                                let new_id = table.register(Function::Builtin(closure));
                                Ok(encode_function_id(new_id))
                            }
                        }
                    }
                } else {
                    Err(EvalError::TypeError(format!("Cannot apply a non-function value: {}", func_val)))
                }
            }
            
            Term::If(cond, then_branch, else_branch) => {
                let condition_val = cond.eval(env, table)?;
                let then_val = then_branch.eval(env, table)?;
                let else_val = else_branch.eval(env, table)?;
                
                let weight = condition_val.max(0.0).min(1.0);
                Ok(weight * then_val + (1.0 - weight) * else_val)
            }
        }
    }
}

// --- Builtin Logic Helpers ---

fn get_builtin_arity(op: &str) -> Result<usize, EvalError> {
    match op {
        // Unary
        "neg" | "abs" | "sqrt" | "fuzzy_not" => Ok(1),
        // Binary
        "+" | "-" | "*" | "/" | "==" | "<" | ">" | "<=" | ">=" | "fuzzy_and" | "fuzzy_or" | "min" | "max" => Ok(2),
        _ => Err(EvalError::TypeError(format!("Unknown builtin: {}", op))),
    }
}

fn execute_builtin(op: &str, args: &[f64]) -> Result<f64, EvalError> {
    match op {
        "neg" => Ok(-args[0]),
        "abs" => Ok(args[0].abs()),
        "sqrt" => Ok(if args[0] < 0.0 { f64::NAN } else { args[0].sqrt() }),
        "fuzzy_not" => Ok(1.0 - args[0].max(0.0).min(1.0)),

        "+" => Ok(args[0] + args[1]),
        "-" => Ok(args[0] - args[1]),
        "*" => Ok(args[0] * args[1]),
        "/" => Ok(if args[1] == 0.0 { f64::INFINITY } else { args[0] / args[1] }),
        "==" => Ok(fuzzy_eq(args[0], args[1])),
        "<" => Ok(if args[0] < args[1] { 1.0 } else { 0.0 }),
        ">" => Ok(if args[0] > args[1] { 1.0 } else { 0.0 }),
        "<=" => Ok(if args[0] <= args[1] { 1.0 } else { 0.0 }),
        ">=" => Ok(if args[0] >= args[1] { 1.0 } else { 0.0 }),
        "fuzzy_and" => Ok(args[0] * args[1]),
        "fuzzy_or" => Ok(args[0] + args[1] - (args[0] * args[1])),
        "min" => Ok(args[0].min(args[1])),
        "max" => Ok(args[0].max(args[1])),
        
        _ => unreachable!(), // Should be caught by arity check
    }
}

// Fuzzy equality with exponential decay
pub fn fuzzy_eq(x: f64, y: f64) -> f64 {
    let scale_factor = 1.0;
    (-((x - y).abs() / scale_factor)).exp()
}

// --- The Parser ---
#[derive(Debug)]
pub enum ParseError {
    UnexpectedChar(char),
    UnexpectedEnd,
    InvalidNumber(String),
    InvalidSyntax(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedChar(c) => write!(f, "Unexpected character: {}", c),
            ParseError::UnexpectedEnd => write!(f, "Unexpected end of input"),
            ParseError::InvalidNumber(s) => write!(f, "Invalid number: {}", s),
            ParseError::InvalidSyntax(s) => write!(f, "Invalid syntax: {}", s),
        }
    }
}

pub struct Parser {
    input: Vec<char>,
    pos: usize,
}

impl Parser {
    pub fn new(input: &str) -> Self { Parser { input: input.chars().collect(), pos: 0 } }
    fn current_char(&self) -> Option<char> { self.input.get(self.pos).copied() }
    fn peek_char(&self) -> Option<char> { self.input.get(self.pos + 1).copied() }
    fn advance(&mut self) { if self.pos < self.input.len() { self.pos += 1; } }
    fn skip_whitespace(&mut self) { while let Some(c) = self.current_char() { if c.is_whitespace() { self.advance(); } else { break; } } }
    
    pub fn parse(&mut self) -> Result<Term, ParseError> {
        self.skip_whitespace();
        let term = self.parse_term()?;
        self.skip_whitespace();
        if self.current_char().is_some() {
            Err(ParseError::UnexpectedChar(self.current_char().unwrap()))
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
                if word == "if" {
                    self.parse_if()
                } else if word == "let" {
                    self.parse_let()
                } else {
                    self.parse_identifier()
                }
            }
            Some(_) => self.parse_identifier(), // For operators like '+'
            None => Err(ParseError::UnexpectedEnd),
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
        if self.input.get(self.pos..self.pos + keyword.len()).map_or(false, |s| s.iter().collect::<String>() == keyword) {
            // Check for word boundary
            if self.input.get(self.pos + keyword.len()).map_or(true, |c| !c.is_alphanumeric()) {
                self.pos += keyword.len();
                return Ok(());
            }
        }
        Err(ParseError::InvalidSyntax(format!("Expected keyword '{}'", keyword)))
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
        self.advance(); // consume '('
        let mut terms = Vec::new();
        loop {
            self.skip_whitespace();
            if self.current_char() == Some(')') { self.advance(); break; }
            if self.current_char().is_none() { return Err(ParseError::UnexpectedEnd); }
            terms.push(self.parse_term()?);
        }
        if terms.is_empty() { return Err(ParseError::InvalidSyntax("Empty application () is not allowed".to_string())); }
        let mut app = terms.remove(0);
        for term in terms { app = Term::App(Box::new(app), Box::new(term)); }
        Ok(app)
    }

    fn parse_lambda(&mut self) -> Result<Term, ParseError> {
        self.advance(); // consume 'λ' or '\'
        self.skip_whitespace();
        let param = self.parse_identifier_string()?;
        self.skip_whitespace();
        if self.current_char() != Some('.') { return Err(ParseError::InvalidSyntax("Expected '.' in lambda".to_string())); }
        self.advance(); // consume '.'
        let body = self.parse_term()?;
        Ok(Term::Lam(param, Box::new(body)))
    }

    fn parse_number(&mut self) -> Result<Term, ParseError> {
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
            .map_err(|_| ParseError::InvalidNumber(s))
    }

    fn parse_identifier(&mut self) -> Result<Term, ParseError> {
        let name = self.parse_identifier_string()?;
        match name.as_str() {
            "neg" | "abs" | "sqrt" | "fuzzy_not" | "+" | "-" | "*" | "/" | "==" | "<" | ">" | "<=" | ">=" | "fuzzy_and" | "fuzzy_or" | "min" | "max" => Ok(Term::Builtin(name)),
            _ => Ok(Term::Var(name)),
        }
    }

    fn parse_identifier_string(&mut self) -> Result<String, ParseError> {
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
            if c.is_alphanumeric() || c == '_' {
                name.push(c);
                self.advance();
            } else {
                break;
            }
        }
        if name.is_empty() { Err(ParseError::InvalidSyntax("Expected an identifier".to_string())) } else { Ok(name) }
    }
}

// Convenience function for parsing
pub fn parse(input: &str) -> Result<Term, ParseError> {
    Parser::new(input).parse()
}

// REPL helper function to orchestrate parsing and evaluation.
fn process_input(
    input: &str,
    table: &mut FunctionTable,
    global_env_map: &mut HashMap<String, f64>,
) -> Result<f64, String> {
    // 1. Parse
    let term = match parse(input) {
        Ok(t) => t,
        Err(e) => return Err(format!("Parse error: {}", e)),
    };
    println!("Parsed: {}", term);

    // 2. Dispatch to the correct handler
    let eval_result = if let Term::Let(name, value, body) = term {
        handle_let_definition(&name, *value, *body, table, global_env_map)
    } else if let Term::LetRec(name, value, body) = term {
        handle_let_rec_definition(&name, *value, *body, table, global_env_map)
    } else {
        // It's a standard expression.
        let eval_env = Rc::new(global_env_map.clone());
        term.eval(&eval_env, table)
    };

    // 3. Format any potential evaluation error
    eval_result.map_err(|e| format!("Eval error: {}", e))
}

// New helper specifically for non-recursive let definitions.
fn handle_let_definition(
    name: &str,
    value: Term,
    body: Term,
    table: &mut FunctionTable,
    global_env_map: &mut HashMap<String, f64>,
) -> Result<f64, EvalError> {
    let eval_env = Rc::new(global_env_map.clone());
    let value_val = value.eval(&eval_env, table)?;

    // Add the new binding to the global environment.
    global_env_map.insert(name.to_string(), value_val);

    // Evaluate the body in the now-updated global environment.
    let body_env = Rc::new(global_env_map.clone());
    body.eval(&body_env, table)
}

// REPL helper specifically for recursive let rec definitions.
fn handle_let_rec_definition(
    name: &str,
    value: Term,
    body: Term,
    table: &mut FunctionTable,
    global_env_map: &mut HashMap<String, f64>,
) -> Result<f64, EvalError> {
    let lam_result = match value {
        Term::Lam(p, b) => Ok(Term::Lam(p, b)),
        _ => Err(EvalError::TypeError(
            "let rec expression must be a lambda".to_string(),
        )),
    };

    lam_result.and_then(|lam| {
        // Create a temporary environment with a placeholder for the function itself.
        let mut temp_env_map = global_env_map.clone();
        temp_env_map.insert(name.to_string(), f64::NAN); // Placeholder
        let temp_env = Rc::new(temp_env_map);
        
        // Evaluate the lambda to create the closure.
        lam.eval(&temp_env, table)
    }).and_then(|func_val| {
        // Decode the function ID.
        match decode_function_id(func_val) {
            Some(id) => {
                // Backpatch the closure's environment to be truly recursive.
                if let Some(Function::User(closure)) = table.get_mut(id) {
                    if let Some(env_map) = Rc::get_mut(&mut closure.env) {
                        env_map.insert(name.to_string(), func_val);
                    } else {
                        let mut new_map = closure.env.as_ref().clone();
                        new_map.insert(name.to_string(), func_val);
                        closure.env = Rc::new(new_map);
                    }
                }

                // Update the global environment with the final, correct function value.
                global_env_map.insert(name.to_string(), func_val);
                let body_env = Rc::new(global_env_map.clone());

                // Finally, evaluate the body.
                body.eval(&body_env, table)
            }
            None => Err(EvalError::TypeError(
                "let rec value did not evaluate to a function".into(),
            )),
        }
    })
}

// Simple REPL
pub fn repl() {
    println!("FloatLambda REPL");
    println!("Enter expressions, 'quit', or ':examples'");

    let mut table = FunctionTable::new();
    let mut global_env_map = HashMap::new();
    let mut last_result = 0.0;

    loop {
        // --- Garbage Collection ---
        let mut roots: Vec<f64> = global_env_map.values().copied().collect();
        roots.push(last_result);
        table.collect(&roots);

        // --- Read Input ---
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input_str = input.trim();

        // --- Process Commands ---
        if input_str == "quit" || input_str == "exit" { break; }
        if input_str.is_empty() { continue; }
        if input_str == ":examples" {
            show_examples();
            continue;
        }

        // --- Evaluate and Print Result ---
        match process_input(input_str, &mut table, &mut global_env_map) {
            Ok(result) => {
                last_result = result; // Update last result on success
                if let Some(id) = decode_function_id(result) {
                    println!("Result: Function<{}> ({} functions alive)", id, table.functions.iter().filter(|f| f.is_some()).count());
                } else {
                    println!("Result: {} ({} functions alive)", result, table.functions.iter().filter(|f| f.is_some()).count());
                }
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

    let mut table = FunctionTable::new();
    let mut global_env_map = HashMap::new();

    // The existing process_input function can be reused here.
    let result = process_input(&content, &mut table, &mut global_env_map)?;

    // Print the final result of the script.
    print_result(result, &table);
    Ok(())
}

/// Helper function to print a result value, checking if it's a function.
/// This avoids duplicating logic between the REPL and the script runner.
fn print_result(result: f64, _table: &FunctionTable) {
    if let Some(id) = decode_function_id(result) {
        println!("Result: Function<{}>", id);
    } else {
        println!("Result: {}", result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to run evaluation
    fn eval_ok(input: &str) -> f64 {
        let term = parse(input).unwrap();
        let mut table = FunctionTable::new(); // Each test gets a fresh function table
        term.eval(&Rc::new(HashMap::new()), &mut table).unwrap()
    }

    fn eval_err(input: &str) -> EvalError {
        let term = parse(input).unwrap();
        let mut table = FunctionTable::new();
        term.eval(&Rc::new(HashMap::new()), &mut table).unwrap_err()
    }

    // --- Original Tests (Re-integrated and Adjusted for New Semantics) ---

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
        assert!(decode_function_id(result).is_some()); // It decodes to a valid ID
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
}