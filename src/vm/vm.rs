use std::collections::HashMap;

use crate::memory::{decode_heap_pointer, encode_heap_pointer, Heap, HeapObject, NIL_VALUE};
use crate::vm::compiler::{compile, CompileError};
use crate::vm::function::Function;
use crate::vm::opcode::OpCode;

#[derive(Debug)]
pub enum InterpretError {
    Compile(CompileError),
    Runtime(String),
}

/// Represents a single ongoing function call.
#[derive(Debug)]
struct CallFrame {
    /// Heap ID of the Function object being executed.
    func_id: u64,
    /// The instruction pointer for this call, pointing into the function's chunk.
    ip: usize,
    /// The index into the VM's main value stack where this function's locals start.
    stack_slot: usize,
}

/// The Virtual Machine.
pub struct VM<'a> {
    heap: &'a mut Heap,
    frames: Vec<CallFrame>,
    stack: Vec<f64>,
    globals: HashMap<String, f64>,
}

/// The main entry point to run code. It handles parsing, compiling, and execution.
pub fn interpret(source: &str, heap: &mut Heap) -> Result<f64, InterpretError> {
    let term = crate::parser::parse(source)
        .map_err(|_| InterpretError::Compile(CompileError::UnsupportedExpression("Parse Error".to_string())))?; // Basic error mapping for now
    let main_chunk = compile(&term).map_err(InterpretError::Compile)?;
    let main_func = Function {
        arity: 0,
        chunk: main_chunk,
        name: "<script>".to_string(),
    };

    let main_id = heap.register(HeapObject::Function(main_func));
    let mut vm = VM::new(heap);
    vm.stack.push(encode_heap_pointer(main_id)); // The function itself is at slot 0.
    vm.call_value(encode_heap_pointer(main_id), 0)?; // Call the top-level script as a function.

    vm.run()
}

impl<'a> VM<'a> {
    fn new(heap: &'a mut Heap) -> Self {
        VM {
            heap,
            frames: Vec::with_capacity(64),
            stack: Vec::with_capacity(256),
            globals: HashMap::new(),
        }
    }

    /// The main execution loop of the VM.
    fn run(&mut self) -> Result<f64, InterpretError> {
        loop {
            // Get necessary info from the current frame.
            let frame = self.frames.last().unwrap();
            let ip = frame.ip;
            let func_id = frame.func_id;

            // --- Read instruction without holding a long borrow ---
            let instruction = { // Scoped to release the borrow immediately
                let func = match self.heap.get(func_id) {
                    Some(HeapObject::Function(f)) => f,
                    _ => return Err(InterpretError::Runtime("CallFrame points to a non-function heap object.".to_string())),
                };
                func.chunk.code[ip]
            };
            
            // Advance the IP for the *next* iteration
            self.frames.last_mut().unwrap().ip += 1;

            let op = OpCode::from(instruction);
            match op {
                OpCode::OpReturn => {
                    let result = self.pop_stack()?;
                    let frame = self.frames.pop().unwrap();
                    if self.frames.is_empty() {
                        return Ok(result); // Final return from the script.
                    }
                    // Discard the returning function's stack window and push its result.
                    self.stack.truncate(frame.stack_slot);
                    self.stack.push(result);
                }
                OpCode::OpConstant => {
                    let const_idx = self.read_byte() as usize;
                    // We need to get the chunk again for this opcode
                    let func = match self.heap.get(func_id).unwrap() {
                        HeapObject::Function(f) => f,
                        _ => unreachable!(),
                    };
                    let constant = func.chunk.constants[const_idx];
                    self.stack.push(constant);
                }
                OpCode::OpNil => self.stack.push(NIL_VALUE),
                OpCode::OpTrue => self.stack.push(1.0),
                OpCode::OpFalse => self.stack.push(0.0),
                OpCode::OpNegate => {
                    let val = self.pop_stack()?;
                    self.stack.push(-val);
                }
                OpCode::OpNot => {
                    let val = self.pop_stack()?;
                    self.stack.push(if val == 0.0 { 1.0 } else { 0.0 });
                }
                OpCode::OpAdd | OpCode::OpSubtract | OpCode::OpMultiply | OpCode::OpDivide => {
                    let b = self.pop_stack()?;
                    let a = self.pop_stack()?;
                    match op {
                        OpCode::OpAdd => self.stack.push(a + b),
                        OpCode::OpSubtract => self.stack.push(a - b),
                        OpCode::OpMultiply => self.stack.push(a * b),
                        OpCode::OpDivide => self.stack.push(if b == 0.0 { f64::INFINITY } else { a / b }),
                        _ => unreachable!(),
                    }
                }
                OpCode::OpGreater | OpCode::OpLess | OpCode::OpEqual => {
                    let b = self.pop_stack()?;
                    let a = self.pop_stack()?;
                    let result = match op {
                        OpCode::OpGreater => a > b,
                        OpCode::OpLess => a < b,
                        OpCode::OpEqual => a.to_bits() == b.to_bits(),
                        _ => unreachable!(),
                    };
                    self.stack.push(if result { 1.0 } else { 0.0 });
                }
                OpCode::OpGetGlobal => {
                    let name_idx = self.read_byte() as usize;
                    // Re-fetch the chunk here as well
                    let func = match self.heap.get(func_id).unwrap() {
                        HeapObject::Function(f) => f,
                        _ => unreachable!(),
                    };
                    let name = &func.chunk.names[name_idx];
                    match self.globals.get(name) {
                        Some(&val) => self.stack.push(val),
                        None => return Err(InterpretError::Runtime(format!("Undefined global variable '{}'.", name))),
                    }
                }
                OpCode::OpDefineGlobal => {
                    let name_idx = self.read_byte() as usize;
                    let name = { // Scoped borrow and clone
                        let func = match self.heap.get(func_id).unwrap() {
                             HeapObject::Function(f) => f, _ => unreachable!(),
                        };
                        func.chunk.names[name_idx].clone()
                    };
                    let val = self.pop_stack()?;
                    self.globals.insert(name, val); // Use cloned name
                }
                OpCode::OpJump => {
                    let offset = self.read_short();
                    self.frames.last_mut().unwrap().ip += offset;
                }
                OpCode::OpJumpIfFalse => {
                    let offset = self.read_short();
                    if let Some(&val) = self.stack.last() {
                        if val == 0.0 || val == NIL_VALUE {
                            self.frames.last_mut().unwrap().ip += offset;
                        }
                    }
                    self.pop_stack()?;
                }
                _ => return Err(InterpretError::Runtime(format!("Unimplemented opcode {:?}", op))),
            }
        }
    }

    fn call_value(&mut self, func_val: f64, arg_count: usize) -> Result<(), InterpretError> {
        let func_id = match decode_heap_pointer(func_val) {
            Some(id) => id,
            None => return Err(InterpretError::Runtime("Can only call functions.".to_string())),
        };

        match self.heap.get(func_id) {
            Some(HeapObject::Function(func)) => {
                if arg_count != func.arity {
                    return Err(InterpretError::Runtime(format!("Expected {} arguments but got {}.", func.arity, arg_count)));
                }
                let frame = CallFrame {
                    func_id,
                    ip: 0,
                    stack_slot: self.stack.len() - arg_count - 1,
                };
                self.frames.push(frame);
                Ok(())
            }
            _ => Err(InterpretError::Runtime("Can only call functions.".to_string())),
        }
    }

    fn read_byte(&mut self) -> u8 {
        let frame = self.frames.last_mut().unwrap();
        let func = match self.heap.get(frame.func_id).unwrap() {
            HeapObject::Function(f) => f,
            _ => panic!("Invalid state in read_byte"),
        };
        let byte = func.chunk.code[frame.ip];
        frame.ip += 1;
        byte
    }

    fn read_short(&mut self) -> usize {
        let frame = self.frames.last_mut().unwrap();
        let func = match self.heap.get(frame.func_id).unwrap() {
            HeapObject::Function(f) => f,
            _ => panic!("Invalid state in read_short"),
        };
        frame.ip += 2;
        let high = func.chunk.code[frame.ip - 2] as usize;
        let low = func.chunk.code[frame.ip - 1] as usize;
        (high << 8) | low
    }

    fn pop_stack(&mut self) -> Result<f64, InterpretError> {
        self.stack.pop().ok_or_else(|| InterpretError::Runtime("Stack underflow.".to_string()))
    }
}