// src/error.rs

use std::fmt;

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