// src/vm/compiler.rs

use crate::ast::Term;
use crate::memory::{encode_heap_pointer, Heap, HeapObject};
use crate::vm::chunk::Chunk;
use crate::vm::function::Function;
use crate::vm::natives;
use crate::vm::opcode::OpCode;
use crate::ParseError;

#[derive(Debug)]
pub enum CompileError {
    UnsupportedExpression(String),
    TooManyConstants,
    Parse(ParseError),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::UnsupportedExpression(msg) => write!(f, "Unsupported expression: {}", msg),
            CompileError::TooManyConstants => write!(f, "Too many constants in chunk."),
            CompileError::Parse(e) => write!(f, "{}", e), 
        }
    }
}

#[derive(Debug)]
struct Local {
    name: String,
    depth: usize,
    is_captured: bool,
}

#[derive(Debug, Clone, Copy)]
struct CompilerUpvalue {
    index: u8,
    is_local: bool,
}

struct Compiler {
    enclosing: Option<Box<Compiler>>,
    function: Function,
    locals: Vec<Local>,
    upvalues: Vec<CompilerUpvalue>,
    scope_depth: usize,
}

pub fn compile(term: &Term, heap: &mut Heap) -> Result<Function, CompileError> {
    let mut compiler = Compiler::new(None, "<script>".to_string(), 0);
    compiler.compile_term(term, heap, true)?;
    Ok(compiler.end_compiler())
}

fn is_blendable(term: &Term) -> bool {
    match term {
        // Simple values are always blendable.
        Term::Float(_) | Term::Builtin(_) => true,
        Term::Var(_) => true,

        Term::App(f, a) => {
            // An application of a user function (a Var) is NOT considered blendable
            // because this is the primary case for tail calls.
            if matches!(&**f, Term::Var(_)) {
                return false;
            }
            // Special case: lambda applications ARE blendable even though lambdas aren't
            if matches!(&**f, Term::Lam(_, _)) {
                return is_blendable(a);  // Only check the argument
            }

            // A call to cons produces a pointer, so it's not blendable.
            if let Term::Builtin(op) = &**f {
                if op == "cons" { return false; }
            }
            // An application of a curried builtin, like ((+ 1) 2), is blendable if its parts are.
            // This recursively checks down the chain of applications.
            is_blendable(f) && is_blendable(a)
        }

        // A let is blendable if its body is.
        Term::Let(_, _, body) => is_blendable(body),

        // An if is blendable if all its parts are.
        Term::If(cond, t, e) => is_blendable(cond) && is_blendable(t) && is_blendable(e),

        // These constructs produce pointers or nil, so they are not blendable.
        Term::Lam(_, _) | Term::LetRec(_, _, _) | Term::Nil => false,
    }
}

impl Compiler {
    fn new(enclosing: Option<Box<Compiler>>, name: String, arity: usize) -> Self {
        Compiler {
            enclosing,
            function: Function { name, arity, ..Default::default() },
            locals: vec![Local { name: "".to_string(), depth: 0, is_captured: false }],
            upvalues: Vec::with_capacity(256),
            scope_depth: 0,
        }
    }

    fn compile_term(&mut self, term: &Term, heap: &mut Heap, is_tail: bool) -> Result<(), CompileError> {
        match term {
            Term::Float(n) => self.emit_constant(*n),
            Term::Nil => self.emit_opcode(OpCode::OpNil),
            Term::Builtin(op) => self.compile_builtin(op, 0)?,
            Term::Lam(p, b) => self.compile_lambda(p, b, heap)?,
            Term::App(f, a) => self.compile_app(f, a, heap, is_tail)?,
            Term::If(c, t, e) => self.compile_if(c, t, e, heap, is_tail)?,
            Term::Let(n, v, b) => self.compile_let(n, v, b, heap, is_tail)?,
            Term::LetRec(n, v, b) => self.compile_let_rec(n, v, b, heap, is_tail)?,
            Term::Var(name) => self.compile_variable(name)?,
        }
        Ok(())
    }

    fn compile_lambda(&mut self, param: &String, body: &Term, heap: &mut Heap) -> Result<(), CompileError> {
        let current_compiler = std::mem::replace(self, Compiler::new(None, "".to_string(), 0));
        let mut func_compiler = Compiler::new(Some(Box::new(current_compiler)), format!("<lambda:{}>", param), 1);
        
        func_compiler.begin_scope();
        func_compiler.add_local(param.clone());
        func_compiler.compile_term(body, heap, true)?;
        let mut function = func_compiler.end_compiler();
        let upvalues = func_compiler.upvalues;
        function.upvalue_count = upvalues.len();
        
        *self = *func_compiler.enclosing.unwrap();

        let func_id = heap.register(HeapObject::Function(function));
        let func_val = encode_heap_pointer(func_id);
        let const_idx = self.add_f64_constant(func_val);

        self.emit_opcode(OpCode::OpClosure);
        self.emit_byte(const_idx as u8);
        for upvalue in upvalues {
            self.emit_byte(if upvalue.is_local { 1 } else { 0 });
            self.emit_byte(upvalue.index);
        }
        Ok(())
    }

    fn compile_variable(&mut self, name: &str) -> Result<(), CompileError> {
        if let Some(index) = self.resolve_local(name)? {
            self.emit_opcode(OpCode::OpGetLocal); self.emit_byte(index as u8);
        } else if let Some(index) = self.resolve_upvalue(name)? {
            self.emit_opcode(OpCode::OpGetUpvalue); self.emit_byte(index as u8);
        } else {
            let name_idx = self.add_name_constant(name.to_string());
            self.emit_opcode(OpCode::OpGetGlobal); self.emit_byte(name_idx as u8);
        }
        Ok(())
    }

    fn compile_app(&mut self, func: &Term, arg: &Term, heap: &mut Heap, is_tail: bool) -> Result<(), CompileError> {
        let mut args = vec![arg];
        let mut current_func = func;
        while let Term::App(f, a) = current_func {
            args.push(a);
            current_func = f;
        }
        
        if let Term::Builtin(op) = current_func {
            for arg in args.iter().rev() {
                self.compile_term(arg, heap, false)?;
            }
            self.compile_builtin(op, args.len())?;
        } else {
            self.compile_term(func, heap, false)?;
            self.compile_term(arg, heap, false)?;
            
            if is_tail {
                self.emit_opcode(OpCode::OpTailCall);
            } else {
                self.emit_opcode(OpCode::OpCall);
            }
            self.emit_byte(1);
        }
        
        Ok(())
    }
        
    fn compile_let(&mut self, name: &str, value: &Term, body: &Term, heap: &mut Heap, is_tail: bool) -> Result<(), CompileError> {
        self.compile_term(value, heap, false)?; // Compile the value, pushing it onto the stack

        if self.scope_depth > 0 { // If not at the global scope, it's a local
            self.begin_scope();
            self.add_local(name.to_string());
            self.compile_term(body, heap, is_tail)?;
            self.end_scope();
        } else { // It's a global variable.
            let name_idx = self.add_name_constant(name.to_string());
            self.emit_opcode(OpCode::OpDefineGlobal);
            self.emit_byte(name_idx as u8);
            // The body is compiled without creating a new scope, so the global persists.
            self.compile_term(body, heap, is_tail)?;
            // pop the value from the stack after the body has used it, unless it was the last expression.
            if !is_tail {
                self.emit_opcode(OpCode::OpPop);
            }
        }
        Ok(())
    }

    fn compile_let_rec(&mut self, name: &str, value: &Term, body: &Term, heap: &mut Heap, is_tail: bool) -> Result<(), CompileError> {
        let name_idx = self.add_name_constant(name.to_string());

        if self.scope_depth > 0 {
            self.begin_scope();
            self.add_local(name.to_string());
            self.compile_term(value, heap, false)?;
            self.compile_term(body, heap, is_tail)?;
            self.end_scope();
        } else { // It's a global recursive function
            self.compile_term(value, heap, false)?;
            self.emit_opcode(OpCode::OpDefineGlobal);
            self.emit_byte(name_idx as u8);
            self.compile_term(body, heap, is_tail)?;
            if !is_tail {
                self.emit_opcode(OpCode::OpPop);
            }
        }
        Ok(())
    }

    fn compile_if(&mut self, cond: &Term, then_b: &Term, else_b: &Term, heap: &mut Heap, is_tail: bool) -> Result<(), CompileError> {
        // println!("Compiling if: then_blendable={}, else_blendable={}", is_blendable(then_b), is_blendable(else_b));

        if is_blendable(then_b) && is_blendable(else_b) {
            // --- STRATEGY 1: FUZZY BLEND ---
            // println!("Using OpBlend strategy");
            self.compile_term(cond, heap, false)?;
            self.compile_term(then_b, heap, false)?;
            self.compile_term(else_b, heap, false)?;
            self.emit_opcode(OpCode::OpBlend);
        } else {
            // --- STRATEGY 2: JUMP-BASED (TCO-compatible) ---
            // println!("Using jump strategy");
            self.compile_term(cond, heap, false)?;
        
            let else_jump = self.emit_jump(OpCode::OpJumpIfFalse);
            self.emit_opcode(OpCode::OpPop); // Pop condition if true
        
            self.compile_term(then_b, heap, is_tail)?;
        
            let end_jump = self.emit_jump(OpCode::OpJump);
        
            self.patch_jump(else_jump)?;
            self.emit_opcode(OpCode::OpPop); // Pop condition if false
        
            self.compile_term(else_b, heap, is_tail)?;
        
            self.patch_jump(end_jump)?;
        }
        
        Ok(())
    }

    fn compile_builtin(&mut self, op: &str, arg_count: usize) -> Result<(), CompileError> {
        let maybe_opcode = match op {
            "+" => Some(OpCode::OpAdd), "-" => Some(OpCode::OpSubtract), 
            "*" => Some(OpCode::OpMultiply), "/" => Some(OpCode::OpDivide),
            "neg" => Some(OpCode::OpNegate), "not" => Some(OpCode::OpNot), 
            "eq?" => Some(OpCode::OpEqual),      // eq? uses strict equality
            "==" => Some(OpCode::OpFuzzyEqual),  // == uses fuzzy equality
            ">=" => {
                self.emit_opcode(OpCode::OpLess);
                self.emit_opcode(OpCode::OpNot);
                return Ok(());
            }
            "<=" => {
                self.emit_opcode(OpCode::OpGreater);
                self.emit_opcode(OpCode::OpNot);
                return Ok(());
            }
            "div" => Some(OpCode::OpDivInt),
            "rem" => Some(OpCode::OpRem),
            "<" => Some(OpCode::OpLess), 
            ">" => Some(OpCode::OpGreater),
            "cons" => Some(OpCode::OpCons), "car" => Some(OpCode::OpCar), "cdr" => Some(OpCode::OpCdr),
            _ => None,
        };

        if let Some(opcode) = maybe_opcode {
            self.emit_opcode(opcode);
            return Ok(());
        }

        if let Some((index, arity)) = natives::NATIVE_MAP.get(op) {
            if arg_count != *arity {
                return Err(CompileError::UnsupportedExpression(
                    format!("Native function '{}' expects {} args, but got {}.", op, arity, arg_count)
                ));
            }
            self.emit_opcode(OpCode::OpNative);
            self.emit_byte(*index);
            return Ok(());
        }

        Err(CompileError::UnsupportedExpression(format!("Unknown builtin function '{}'.", op)))
    }

    fn resolve_local(&self, name: &str) -> Result<Option<usize>, CompileError> {
        for (i, local) in self.locals.iter().enumerate().rev() {
            if local.name == name { return Ok(Some(i)); }
        }
        Ok(None)
    }

    fn resolve_upvalue(&mut self, name: &str) -> Result<Option<usize>, CompileError> {
        if self.enclosing.is_none() { return Ok(None); }
        if let Some(local_idx) = self.enclosing.as_mut().unwrap().resolve_local(name)? {
            self.enclosing.as_mut().unwrap().locals[local_idx].is_captured = true;
            return Ok(Some(self.add_upvalue(local_idx as u8, true)?));
        }
        if let Some(upvalue_idx) = self.enclosing.as_mut().unwrap().resolve_upvalue(name)? {
            return Ok(Some(self.add_upvalue(upvalue_idx as u8, false)?));
        }
        Ok(None)
    }

    fn add_upvalue(&mut self, index: u8, is_local: bool) -> Result<usize, CompileError> {
        for (i, upvalue) in self.upvalues.iter().enumerate() {
            if upvalue.index == index && upvalue.is_local == is_local { return Ok(i); }
        }
        self.upvalues.push(CompilerUpvalue { index, is_local });
        Ok(self.upvalues.len() - 1)
    }

    fn end_compiler(&mut self) -> Function {
        self.emit_opcode(OpCode::OpReturn);
        std::mem::take(&mut self.function)
    }
    
    fn begin_scope(&mut self) { self.scope_depth += 1; }

    fn end_scope(&mut self) {
        self.scope_depth -= 1;
        while !self.locals.is_empty() && self.locals.last().unwrap().depth > self.scope_depth {
            if self.locals.last().unwrap().is_captured {
                self.emit_opcode(OpCode::OpCloseUpvalue);
            } else {
                self.emit_opcode(OpCode::OpPop);
            }
            self.locals.pop();
        }
    }

    fn add_local(&mut self, name: String) {
        self.locals.push(Local { name, depth: self.scope_depth, is_captured: false });
    }

    fn current_chunk(&mut self) -> &mut Chunk { &mut self.function.chunk }
    
    fn emit_byte(&mut self, byte: u8) { self.current_chunk().write(byte, 0); }

    fn emit_opcode(&mut self, op: OpCode) { self.current_chunk().write_opcode(op, 0); }

    fn emit_constant(&mut self, value: f64) {
        let const_idx = self.add_f64_constant(value);
        self.emit_opcode(OpCode::OpConstant);
        self.emit_byte(const_idx as u8);
    }
    
    fn add_f64_constant(&mut self, value: f64) -> usize { self.current_chunk().add_constant(value) }
    
    fn add_name_constant(&mut self, name: String) -> usize { self.current_chunk().add_name(name) }

    fn emit_jump(&mut self, op: OpCode) -> usize {
        self.emit_opcode(op);
        self.emit_byte(0xff); self.emit_byte(0xff);
        self.current_chunk().code.len() - 2
    }
    
    fn patch_jump(&mut self, offset_pos: usize) -> Result<(), CompileError> {
        let jump = self.current_chunk().code.len() - offset_pos - 2;
        if jump > u16::MAX as usize { return Err(CompileError::UnsupportedExpression("Jump too large.".to_string())); }
        self.current_chunk().code[offset_pos] = ((jump >> 8) & 0xff) as u8;
        self.current_chunk().code[offset_pos + 1] = (jump & 0xff) as u8;
        Ok(())
    }
}
