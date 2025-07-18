// src/vm/compiler.rs

use crate::ast::Term;
use crate::memory::{encode_heap_pointer, Heap, HeapObject};
use crate::vm::chunk::Chunk;
use crate::vm::function::Function;
use crate::vm::opcode::OpCode;

// ... (CompileError and Local structs remain the same) ...
#[derive(Debug)]
pub enum CompileError {
    UnsupportedExpression(String),
    TooManyConstants,
    VariableNotFound(String),
    ParseError,
}

#[derive(Debug)]
struct Local {
    name: String,
    depth: usize,
}

/// The main entry point for compilation.
pub fn compile(term: &Term, heap: &mut Heap) -> Result<Function, CompileError> {
    let mut compiler = Compiler::new("<script>".to_string(), 0);
    compiler.compile_term(term, heap)?;
    compiler.emit_return();
    Ok(compiler.function)
}

struct Compiler {
    function: Function,
    locals: Vec<Local>,
    scope_depth: usize,
}

impl Compiler {
    fn new(name: String, arity: usize) -> Self {
        Compiler {
            function: Function { name, arity, ..Default::default() },
            locals: vec![Local { name: "".to_string(), depth: 0 }],
            scope_depth: 0,
        }
    }

    fn compile_term(&mut self, term: &Term, heap: &mut Heap) -> Result<(), CompileError> {
        match term {
            Term::Float(n) => self.emit_constant(*n),
            Term::Nil => self.emit_opcode(OpCode::OpNil),
            Term::Builtin(op) => self.compile_builtin(op)?,
            Term::Var(name) => {
                if let Some(index) = self.resolve_local(name)? {
                    self.emit_opcode(OpCode::OpGetLocal);
                    self.emit_byte(index as u8);
                } else {
                    let name_idx = self.add_name_constant(name.clone());
                    self.emit_opcode(OpCode::OpGetGlobal);
                    self.emit_byte(name_idx as u8);
                }
            }
            Term::Let(name, value, body) => {
                self.compile_term(value, heap)?;
                if self.scope_depth > 0 {
                    self.add_local(name.clone());
                    self.mark_initialized();
                } else {
                    let name_idx = self.add_name_constant(name.clone());
                    self.emit_opcode(OpCode::OpDefineGlobal);
                    self.emit_byte(name_idx as u8);
                }
                self.compile_term(body, heap)?;
            }
            Term::If(condition, then_b, else_b) => self.compile_if(condition, then_b, else_b, heap)?,

            Term::App(func, arg) => {
                let mut args = vec![arg.as_ref()];
                let mut current_func = func.as_ref();

                while let Term::App(f, a) = current_func {
                    args.push(a.as_ref());
                    current_func = f.as_ref();
                }

                // If the root is a builtin, it's a simple operator.
                // Compile the arguments, then the operator itself.
                if let Term::Builtin(op) = current_func {
                    for arg in args.iter().rev() {
                        self.compile_term(arg, heap)?;
                    }
                    self.compile_builtin(op)?;
                } else {
                    // Otherwise, it's a user-defined function call.
                    // Compile the function expression, then the arguments.
                    self.compile_term(current_func, heap)?;
                    for arg in args.iter().rev() {
                        self.compile_term(arg, heap)?;
                    }
                    self.emit_opcode(OpCode::OpCall);
                    self.emit_byte(args.len() as u8);
                }
            }
            
            Term::Lam(param, body) => {
                // Count the arity of the curried function
                let mut params = vec![param.clone()];
                let mut current_body = body;
                while let Term::Lam(p, b) = &**current_body {
                    params.push(p.clone());
                    current_body = b;
                }
                
                let mut func_compiler = Compiler::new(format!("<lambda:{}>", params.join(", ")), params.len());
                func_compiler.begin_scope();

                // Add all parameters as the first local variables.
                for p_name in &params {
                    func_compiler.add_local(p_name.clone());
                    func_compiler.mark_initialized();
                }

                func_compiler.compile_term(current_body, heap)?;
                func_compiler.emit_return();
                
                let function = func_compiler.function;
                let func_id = heap.register(HeapObject::Function(function));
                let func_val = encode_heap_pointer(func_id);

                let const_idx = self.add_f64_constant(func_val);
                self.emit_opcode(OpCode::OpConstant);
                self.emit_byte(const_idx as u8);
            }

            _ => return Err(CompileError::UnsupportedExpression(format!("{:?}", term))),
        }
        Ok(())
    }
    
    fn compile_if(&mut self, cond: &Term, then_b: &Term, else_b: &Term, heap: &mut Heap) -> Result<(), CompileError> {
        // 1. Compile the condition.
        self.compile_term(cond, heap)?;
        
        // 2. Emit jump to the 'else' branch if condition is false.
        let else_jump = self.emit_jump(OpCode::OpJumpIfFalse);
        
        // 3. Compile the 'then' branch. Pop the condition first.
        self.emit_opcode(OpCode::OpPop); 
        self.compile_term(then_b, heap)?;
        
        // 4. Emit jump to skip the 'else' branch.
        let end_jump = self.emit_jump(OpCode::OpJump);
        
        // 5. Patch the jump to the 'else' branch.
        self.patch_jump(else_jump)?;
        
        // 6. Compile the 'else' branch. Pop the condition first.
        self.emit_opcode(OpCode::OpPop);
        self.compile_term(else_b, heap)?;
        
        // 7. Patch the jump that skips the 'else' branch.
        self.patch_jump(end_jump)?;
        
        Ok(())
    }

    fn compile_builtin(&mut self, op: &str) -> Result<(), CompileError> {
        let op_code = match op {
            "+" => OpCode::OpAdd, "-" => OpCode::OpSubtract, "*" => OpCode::OpMultiply, "/" => OpCode::OpDivide,
            "neg" => OpCode::OpNegate, "not" => OpCode::OpNot, "==" => OpCode::OpEqual,
            "<" => OpCode::OpLess, ">" => OpCode::OpGreater,
            _ => return Err(CompileError::UnsupportedExpression(op.to_string())),
        };
        self.emit_opcode(op_code);
        Ok(())
    }
    
    // --- Scope and Variable Helpers ---
    fn begin_scope(&mut self) { self.scope_depth += 1; }
    fn end_scope(&mut self) {
        self.scope_depth -= 1;
        while !self.locals.is_empty() && self.locals.last().unwrap().depth > self.scope_depth {
            self.locals.pop();
            self.emit_opcode(OpCode::OpPop);
        }
    }
    fn add_local(&mut self, name: String) { self.locals.push(Local { name, depth: self.scope_depth }); }
    fn mark_initialized(&mut self) { self.locals.last_mut().unwrap().depth = self.scope_depth; }
    fn resolve_local(&self, name: &str) -> Result<Option<usize>, CompileError> {
        for (i, local) in self.locals.iter().enumerate().rev() {
            if local.name == name { return Ok(Some(i)); }
        }
        Ok(None)
    }

    // --- Bytecode Emitter Helpers ---
    fn current_chunk(&mut self) -> &mut Chunk { &mut self.function.chunk }
    fn emit_byte(&mut self, byte: u8) { self.current_chunk().write(byte, 0); }
    fn emit_opcode(&mut self, op: OpCode) { self.current_chunk().write_opcode(op, 0); }
    fn emit_return(&mut self) {
            // The return value is already on top of the stack. Just return it.
            self.emit_opcode(OpCode::OpReturn);
        }
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
