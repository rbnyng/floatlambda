// src/vm/vm.rs

use crate::memory::NIL_VALUE;
use crate::vm::chunk::Chunk;
use crate::vm::opcode::OpCode;
use std::collections::HashMap; 

// Combined result type for the VM.
#[derive(Debug)]
pub enum InterpretError {
    CompileError, // Placeholder for now
    RuntimeError(String),
}

/// The main entry point to run code through the bytecode backend.
pub fn interpret(chunk: &Chunk) -> Result<f64, InterpretError> {
    let mut vm = VM::new(chunk);
    vm.run()
}

/// The Virtual Machine struct.
struct VM<'a> {
    chunk: &'a Chunk,
    ip: usize, // Instruction Pointer
    stack: Vec<f64>,
    globals: HashMap<String, f64>, 
}

impl<'a> VM<'a> {
    fn new(chunk: &'a Chunk) -> Self {
        VM {
            chunk,
            ip: 0,
            stack: Vec::with_capacity(256), // Pre-allocate some stack space
            globals: HashMap::new(), 
        }
    }

    /// The main execution loop.
    fn run(&mut self) -> Result<f64, InterpretError> {
        loop {
            // Read the instruction at the current pointer
            let instruction = self.chunk.code[self.ip];
            self.ip += 1;

            let op = OpCode::from(instruction);

            match op {
                OpCode::OpReturn => {
                    // The final result of the script is on top of the stack.
                    return Ok(self.stack.pop().unwrap_or(NIL_VALUE));
                }
                OpCode::OpConstant => {
                    let const_idx = self.chunk.code[self.ip] as usize;
                    self.ip += 1;
                    let constant = self.chunk.constants[const_idx];
                    self.stack.push(constant);
                }
                OpCode::OpNil => self.stack.push(NIL_VALUE),
                OpCode::OpTrue => self.stack.push(1.0),
                OpCode::OpFalse => self.stack.push(0.0),
                OpCode::OpNegate => {
                    let val = self.pop_stack()?;
                    self.stack.push(-val);
                }
                OpCode::OpAdd | OpCode::OpSubtract | OpCode::OpMultiply | OpCode::OpDivide => {
                    // Note: Order matters. The right-hand operand is pushed last, so it's popped first.
                    let b = self.pop_stack()?;
                    let a = self.pop_stack()?;
                    match op {
                        OpCode::OpAdd => self.stack.push(a + b),
                        OpCode::OpSubtract => self.stack.push(a - b),
                        OpCode::OpMultiply => self.stack.push(a * b),
                        OpCode::OpDivide => self.stack.push(if b == 0.0 { f64::INFINITY } else { a / b }),
                        _ => unreachable!(), // Should not happen
                    }
                }
                OpCode::OpDefineGlobal => {
                    let name_idx = self.chunk.code[self.ip] as usize;
                    self.ip += 1;
                    let name = &self.chunk.names[name_idx];
                    // The value is on top of the stack.
                    // We use peek because a let expression's value is the result of its body,
                    // not the assignment. The value remains on the stack for the next expression.
                    if let Some(val) = self.stack.last() {
                         self.globals.insert(name.clone(), *val);
                         // After defining, we pop the value off. The result of a let is its body.
                         self.pop_stack()?;
                    } else {
                        return Err(InterpretError::RuntimeError("Stack empty on global define.".to_string()));
                    }
                }
                OpCode::OpGetGlobal => {
                    let name_idx = self.chunk.code[self.ip] as usize;
                    self.ip += 1;
                    let name = &self.chunk.names[name_idx];
                    match self.globals.get(name) {
                        Some(val) => self.stack.push(*val),
                        None => {
                            return Err(InterpretError::RuntimeError(format!(
                                "Undefined global variable '{}'.",
                                name
                            )));
                        }
                    }
                }
                OpCode::OpJump => {
                    let offset = self.read_short();
                    self.ip += offset;
                }
                OpCode::OpJumpIfFalse => {
                    let offset = self.read_short();
                    // Peek at the condition value without popping it yet.
                    if let Some(&val) = self.stack.last() {
                        if val == 0.0 || val == NIL_VALUE {
                            self.ip += offset;
                        }
                    }
                    // The condition value is no longer needed after the check.
                    self.pop_stack()?;
                }
                OpCode::OpGreater | OpCode::OpLess | OpCode::OpEqual => {
                    let b = self.pop_stack()?;
                    let a = self.pop_stack()?;
                    let result = match op {
                        OpCode::OpGreater => a > b,
                        OpCode::OpLess => a < b,
                        OpCode::OpEqual => a.to_bits() == b.to_bits(), // Strict bitwise eq?
                        _ => unreachable!(),
                    };
                    self.stack.push(if result { 1.0 } else { 0.0 });
                }
                _ => {
                    return Err(InterpretError::RuntimeError(format!(
                        "Unknown opcode {:?}",
                        op
                    )))
                }
            }
        }
    }

    // Reads the next two bytes from the chunk as a u16 offset.
    fn read_short(&mut self) -> usize {
        self.ip += 2;
        let high = self.chunk.code[self.ip - 2] as usize;
        let low = self.chunk.code[self.ip - 1] as usize;
        (high << 8) | low
    }

    // Helper to pop from the stack, returning a runtime error on underflow.
    fn pop_stack(&mut self) -> Result<f64, InterpretError> {
        self.stack.pop().ok_or_else(|| {
            InterpretError::RuntimeError("Stack underflow.".to_string())
        })
    }
}