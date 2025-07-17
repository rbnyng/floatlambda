// src/parser.rs

use crate::ast::Term;
use crate::error::{ParseError, ParseErrorKind};

// --- The Parser ---
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
            "print" | "read-char" | "read-line" |
            "length" | "map" | "filter" | "foldl" => Ok(Term::Builtin(name)), 
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