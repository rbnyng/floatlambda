// src/ast.rs

use std::fmt;

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