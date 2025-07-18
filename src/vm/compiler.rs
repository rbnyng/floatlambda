// src/vm/compiler.rs

use crate::ast::Term;
use crate::vm::chunk::Chunk;
use crate::vm::opcode::OpCode;

// A simple error type for compilation problems.
#[derive(Debug)]
pub enum CompileError {
    UnsupportedExpression(String),
    TooManyConstants,
}

/// The main compilation function. It creates a Compiler and runs it.
pub fn compile(term: &Term) -> Result<Chunk, CompileError> {
    let mut compiler = Compiler::new();
    compiler.compile_term(term)?;
    // Every script should end by returning its final value.
    compiler.chunk.write_opcode(OpCode::OpReturn, 0); // Assuming line 0 for now
    Ok(compiler.chunk)
}

/// Manages the state of the compilation process.
struct Compiler {
    chunk: Chunk,
}

impl Compiler {
    fn new() -> Self {
        Compiler {
            chunk: Chunk::new(),
        }
    }

    /// The main recursive function to walk the AST and emit bytecode.
    fn compile_term(&mut self, term: &Term) -> Result<(), CompileError> {
        // We'll need the line number eventually, for now we pass 0.
        let line = 0;

        match term {
            Term::Float(n) => {
                let const_idx = self.chunk.add_constant(*n);
                if const_idx > u8::MAX as usize {
                    return Err(CompileError::TooManyConstants);
                }
                self.chunk.write_opcode(OpCode::OpConstant, line);
                self.chunk.write(const_idx as u8, line);
            }
            Term::Nil => {
                self.chunk.write_opcode(OpCode::OpNil, line);
            }
            Term::App(func, arg) => {
                // For this phase, we only handle simple binary arithmetic.
                // The AST for (+ 1 2) is App(App(Builtin("+"), Float(1)), Float(2))
                if let Term::App(inner_func, first_arg) = &**func {
                    // Compile arguments first, pushing them onto the stack.
                    self.compile_term(first_arg)?;
                    self.compile_term(arg)?;
                    // Then compile the operator itself.
                    self.compile_term(inner_func)?;
                } else {
                    // Handle unary operations like (neg 1).
                    self.compile_term(arg)?;
                    self.compile_term(func)?;
                }
            }
            Term::Builtin(op) => {
                let op_code = match op.as_str() {
                    "+" => OpCode::OpAdd,
                    "-" => OpCode::OpSubtract,
                    "*" => OpCode::OpMultiply,
                    "/" => OpCode::OpDivide,
                    "neg" => OpCode::OpNegate,
                    "==" => OpCode::OpEqual, // Note: We need to define semantics for this later.
                    "<" => OpCode::OpLess,
                    ">" => OpCode::OpGreater,
                    "not" => OpCode::OpNot,
                    _ => return Err(CompileError::UnsupportedExpression(op.clone())),
                };
                self.chunk.write_opcode(op_code, line);
            }
            Term::Let(name, value, body) => {
                // Compile the expression that produces the variable's value.
                self.compile_term(value)?;
                
                // Add the variable name to the name pool.
                let name_idx = self.chunk.add_name(name.clone());
                if name_idx > u8::MAX as usize {
                    return Err(CompileError::TooManyConstants); // Re-use this error for now
                }

                // Emit the instruction to define the global.
                self.chunk.write_opcode(OpCode::OpDefineGlobal, line);
                self.chunk.write(name_idx as u8, line);

                // Now compile the body where this variable is used.
                self.compile_term(body)?;
            }
            Term::Var(name) => {
                let name_idx = self.chunk.add_name(name.clone());
                if name_idx > u8::MAX as usize {
                    return Err(CompileError::TooManyConstants);
                }

                // Emit the instruction to get the global's value.
                self.chunk.write_opcode(OpCode::OpGetGlobal, line);
                self.chunk.write(name_idx as u8, line);
            }
            Term::If(condition, then_branch, else_branch) => {
                // 1. Compile the condition
                self.compile_term(condition)?;
                
                // 2. Emit a JUMP_IF_FALSE with a placeholder offset.
                let then_jump = self.emit_jump(OpCode::OpJumpIfFalse, line);

                // 3. Compile the 'then' branch.
                self.compile_term(then_branch)?;

                // 4. Emit a JUMP to skip over the 'else' branch.
                let else_jump = self.emit_jump(OpCode::OpJump, line);
                
                // 5. Backpatch the 'then_jump' to point to right after the 'else_jump'.
                self.patch_jump(then_jump)?;

                // 6. Compile the 'else' branch.
                self.compile_term(else_branch)?;

                // 7. Backpatch the 'else_jump' to point to the end of the expression.
                self.patch_jump(else_jump)?;
            }
            _ => {
                // We don't support Vars, Lambdas, Ifs, etc., yet.
                return Err(CompileError::UnsupportedExpression(format!("{:?}", term)));
            }
        }
        Ok(())
    }

    /// Emits a jump instruction and a 16-bit placeholder operand.
    /// Returns the position of the placeholder to be patched later.
    fn emit_jump(&mut self, op: OpCode, line: usize) -> usize {
        self.chunk.write_opcode(op, line);
        self.chunk.write(0xff, line); // Placeholder byte 1
        self.chunk.write(0xff, line); // Placeholder byte 2
        self.chunk.code.len() - 2
    }

    /// Calculates the jump offset and patches the bytecode.
    fn patch_jump(&mut self, offset_pos: usize) -> Result<(), CompileError> {
        // -2 to account for the size of the jump offset itself.
        let jump = self.chunk.code.len() - offset_pos - 2;
        if jump > u16::MAX as usize {
            return Err(CompileError::UnsupportedExpression("Jump too large.".to_string()));
        }

        self.chunk.code[offset_pos] = ((jump >> 8) & 0xff) as u8;
        self.chunk.code[offset_pos + 1] = (jump & 0xff) as u8;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn test_compile_simple_arithmetic() {
        // Expression: (neg (+ 1 2))
        // This should compile to:
        // 1. Push 1.0
        // 2. Push 2.0
        // 3. Add
        // 4. Negate
        // 5. Return
        let term = parse("(neg (+ 1 2))").unwrap();
        let chunk = compile(&term).unwrap();

        // Check constants
        assert_eq!(chunk.constants, vec![1.0, 2.0]);

        // Check bytecode sequence
        let expected_code = vec![
            OpCode::OpConstant as u8, 0, // PUSH 1.0
            OpCode::OpConstant as u8, 1, // PUSH 2.0
            OpCode::OpAdd as u8,
            OpCode::OpNegate as u8,
            OpCode::OpReturn as u8,
        ];
        assert_eq!(chunk.code, expected_code);
    }
}