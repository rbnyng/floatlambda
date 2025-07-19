// src/vm/vm.rs

use crate::memory::{decode_heap_pointer, encode_heap_pointer, Heap, HeapObject, NIL_VALUE};
use crate::vm::closure::{Closure as VMClosure, Upvalue};
use crate::vm::compiler::{compile, CompileError};
use crate::vm::natives;
use crate::vm::opcode::OpCode;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug)]
pub enum InterpretError {
    Compile(CompileError),
    Runtime(String),
}

impl std::fmt::Display for InterpretError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpretError::Compile(e) => write!(f, "Compilation Error: {}", e),
            InterpretError::Runtime(msg) => write!(f, "Runtime Error: {}", msg),
        }
    }
}

/// Represents a single ongoing function call.
#[derive(Debug)]
pub struct CallFrame {
    /// Heap ID of the Function object being executed.
    pub closure_id: u64,
    /// The instruction pointer for this call, pointing into the function's chunk.
    pub ip: usize,
    /// The index into the VM's main value stack where this function's locals start.
    pub stack_slot: usize,
}

/// The Virtual Machine.
pub struct VM<'a> {
    pub heap: &'a mut Heap,
    pub frames: Vec<CallFrame>,
    pub stack: Vec<f64>,
    globals: HashMap<String, f64>,
    open_upvalues: Vec<u64>,
}

// Helper function for fuzzy equality
pub fn fuzzy_eq(x: f64, y: f64) -> f64 {
    let diff = (x - y).abs();
    // Use max of absolute values of x and y, but at least 1.0 to avoid inflating the result
    // for small numbers and to prevent division by zero.
    let scale_factor = x.abs().max(y.abs()).max(1.0);
    (-(diff / scale_factor)).exp()
}

/// The main entry point to run code. It handles parsing, compiling, and execution.
pub fn interpret(source: &str, heap: &mut Heap) -> Result<f64, InterpretError> {
    let term = crate::parser::parse(source).map_err(|_| InterpretError::Compile(CompileError::ParseError))?;
    let mut main_func = compile(&term, heap).map_err(InterpretError::Compile)?;
    main_func.name = "<script>".to_string();
    
    let main_id = heap.register(HeapObject::Function(main_func));
    let main_closure = VMClosure { func_id: main_id, upvalues: Rc::new(Vec::new()) };

    let main_closure_id = heap.register(HeapObject::Closure(main_closure));

    let mut vm = VM::new(heap);
    
    // Prime the VM for the first call
    vm.stack.push(encode_heap_pointer(main_closure_id));
    vm.call_value(encode_heap_pointer(main_closure_id), 0)?;

    // Start the main execution loop
    vm.run()
}

impl<'a> VM<'a> {
    pub fn new(heap: &'a mut Heap) -> Self {
        VM {
            heap,
            frames: Vec::with_capacity(64),
            stack: Vec::with_capacity(256),
            globals: HashMap::new(),
            open_upvalues: Vec::new(),
        }
    }

    /// The re-entrant execution loop of the VM.
    /// It runs until the number of call frames drops below what it was when the loop was entered.
    pub fn run(&mut self) -> Result<f64, InterpretError> {
        let frame_count_at_start = self.frames.len();
        if frame_count_at_start == 0 {
            return Err(InterpretError::Runtime("Cannot run VM without a call frame.".to_string()));
        }

        loop {
            let frame = self.frames.last().unwrap();
            let ip = frame.ip;
            let closure_id = frame.closure_id;

            let (func_id, upvalues_rc) = match self.heap.get(closure_id).unwrap() {
                HeapObject::Closure(c) => (c.func_id, c.upvalues.clone()),
                _ => return Err(InterpretError::Runtime("Expected closure on heap.".to_string())),
            };
            let func = match self.heap.get(func_id).unwrap() {
                HeapObject::Function(f) => f,
                _ => return Err(InterpretError::Runtime("Expected function on heap.".to_string())),
            };

            let instruction = func.chunk.code[ip];
            self.frames.last_mut().unwrap().ip += 1;
            let op = OpCode::from(instruction);

            match op {
                OpCode::OpReturn => {
                    let result = self.pop_stack()?;
                    let frame = self.frames.pop().unwrap();
                    self.close_upvalues(frame.stack_slot);

                    // If we have returned from the call that started this `run` invocation,
                    // then we are done with this sub-computation.
                    if self.frames.len() < frame_count_at_start {
                        // The `run` contract is to leave the result on the stack for the native.
                        // However, for cleanliness, we also return it as a Rust value.
                        self.stack.push(result);
                        return Ok(result);
                    }

                    // Otherwise, it was a normal function return within the same `run` context.
                    self.stack.truncate(frame.stack_slot);
                    self.stack.push(result);
                }
                OpCode::OpConstant => {
                    let const_val = self.read_constant()?;
                    self.stack.push(const_val);
                }
                // ... (all other opcodes are the same as before) ...
                OpCode::OpNil => self.stack.push(NIL_VALUE),
                OpCode::OpTrue => self.stack.push(1.0),
                OpCode::OpFalse => self.stack.push(0.0),
                OpCode::OpCons => {
                    let cdr_val = self.pop_stack()?;
                    let car_val = self.pop_stack()?;
                    let pair_id = self.heap.register(HeapObject::Pair(car_val, cdr_val));
                    self.stack.push(encode_heap_pointer(pair_id));
                }
                OpCode::OpCar => {
                    let val = self.pop_stack()?;
                    let pair_id = decode_heap_pointer(val)
                        .ok_or_else(|| InterpretError::Runtime("'car' expects a pair, but got a number or nil.".to_string()))?;
                    match self.heap.get(pair_id) {
                        Some(HeapObject::Pair(car, _)) => self.stack.push(*car),
                        _ => return Err(InterpretError::Runtime("'car' expects a pair.".to_string())),
                    }
                }
                OpCode::OpCdr => {
                    let val = self.pop_stack()?;
                    let pair_id = decode_heap_pointer(val)
                        .ok_or_else(|| InterpretError::Runtime("'cdr' expects a pair, but got a number or nil.".to_string()))?;
                    match self.heap.get(pair_id) {
                        Some(HeapObject::Pair(_, cdr)) => self.stack.push(*cdr),
                        _ => return Err(InterpretError::Runtime("'cdr' expects a pair.".to_string())),
                    }
                }
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
                OpCode::OpFuzzyEqual => {
                    let b = self.pop_stack()?;
                    let a = self.pop_stack()?;
                    self.stack.push(fuzzy_eq(a, b));
                }
                OpCode::OpGetGlobal => {
                    let name_idx = self.read_byte() as usize;
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
                    let name = {
                        let func = match self.heap.get(func_id).unwrap() {
                            HeapObject::Function(f) => f, _ => unreachable!(),
                        };
                        func.chunk.names[name_idx].clone()
                    };
                    let val = self.pop_stack()?;
                    self.globals.insert(name, val);
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
                }
                OpCode::OpPop => {
                    self.pop_stack()?;
                }
                OpCode::OpGetLocal => {
                    let slot = self.read_byte() as usize;
                    let frame = self.frames.last().unwrap();
                    self.stack.push(self.stack[frame.stack_slot + slot]);
                }
                OpCode::OpSetLocal => {
                    let slot = self.read_byte() as usize;
                    let frame = self.frames.last().unwrap();
                    self.stack[frame.stack_slot + slot] = *self.stack.last().unwrap();
                }
                OpCode::OpCall => {
                    let arg_count = self.read_byte() as usize;
                    let func_to_call_val = self.stack[self.stack.len() - 1 - arg_count];
                    self.call_value(func_to_call_val, arg_count)?;
                }
                OpCode::OpTailCall => {
                    let arg_count = self.read_byte() as usize;
                    let func_val = self.stack[self.stack.len() - 1 - arg_count];
                    let new_closure_id = decode_heap_pointer(func_val)
                        .ok_or_else(|| InterpretError::Runtime("Can only call functions.".to_string()))?;

                    let arity = {
                        let new_closure = match self.heap.get(new_closure_id) {
                            Some(HeapObject::Closure(c)) => c,
                            _ => return Err(InterpretError::Runtime("Can only call functions.".to_string())),
                        };
                        let new_func = match self.heap.get(new_closure.func_id) {
                            Some(HeapObject::Function(f)) => f,
                            _ => return Err(InterpretError::Runtime("Closure points to a non-function.".to_string())),
                        };
                        new_func.arity
                    };
                    if arg_count != arity {
                        return Err(InterpretError::Runtime(format!("Expected {} arguments but got {}.", arity, arg_count)));
                    }

                    let stack_slot = self.frames.last().unwrap().stack_slot;
                    self.close_upvalues(stack_slot);

                    for i in 0..=arg_count {
                        self.stack[stack_slot + i] = self.stack[self.stack.len() - 1 - arg_count + i];
                    }
                    self.stack.truncate(stack_slot + arg_count + 1);

                    let current_frame = self.frames.last_mut().unwrap();
                    current_frame.closure_id = new_closure_id;
                    current_frame.ip = 0;
                }
                OpCode::OpClosure => {
                    let func_id = decode_heap_pointer(self.read_constant()?).unwrap();
                    let func_obj = match self.heap.get(func_id).unwrap() {
                        HeapObject::Function(f) => f, _ => unreachable!(),
                    };

                    let mut upvalues = Vec::with_capacity(func_obj.upvalue_count);
                    for _ in 0..func_obj.upvalue_count {
                        let is_local = self.read_byte() == 1;
                        let index = self.read_byte() as usize;
                        if is_local {
                            let location = self.frames.last().unwrap().stack_slot + index;
                            upvalues.push(self.capture_upvalue(location));
                        } else {
                            upvalues.push(upvalues_rc[index]);
                        }
                    }

                    let closure = VMClosure { func_id, upvalues: Rc::new(upvalues) };
                    let closure_id = self.heap.register(HeapObject::Closure(closure));
                    self.stack.push(encode_heap_pointer(closure_id));
                }
                OpCode::OpGetUpvalue => {
                    let slot = self.read_byte() as usize;
                    let upvalue_id = upvalues_rc[slot];
                    let upvalue_obj = self.heap.get(upvalue_id).unwrap();
                    if let HeapObject::Upvalue(up) = upvalue_obj {
                        match up {
                            Upvalue::Open(location) => self.stack.push(self.stack[*location]),
                            Upvalue::Closed(val) => self.stack.push(*val),
                        }
                    } else { unreachable!() }
                }
                OpCode::OpCloseUpvalue => {
                    self.close_upvalues(self.stack.len() - 1);
                    self.pop_stack()?;
                }
                OpCode::OpNative => {
                    let native_index = self.read_byte() as usize;
                    let native = &natives::NATIVES[native_index];
                    (native.func)(self)?;
                }
                // The new calculus opcodes are natives now, no need for new match arms.
                _ => return Err(InterpretError::Runtime(format!("Unimplemented opcode {:?}", op))),
            }
        }
    }

    pub fn call_value(&mut self, func_val: f64, arg_count: usize) -> Result<(), InterpretError> {
        let closure_id = match decode_heap_pointer(func_val) {
            Some(id) => id,
            None => return Err(InterpretError::Runtime("Can only call functions.".to_string())),
        };

        match self.heap.get(closure_id) {
            Some(HeapObject::Closure(closure)) => {
                let func = match self.heap.get(closure.func_id) {
                    Some(HeapObject::Function(f)) => f,
                    _ => return Err(InterpretError::Runtime("Closure points to a non-function.".to_string())),
                };
                if arg_count != func.arity {
                    return Err(InterpretError::Runtime(format!("Expected {} arguments but got {}.", func.arity, arg_count)));
                }
                let frame = CallFrame { closure_id, ip: 0, stack_slot: self.stack.len() - arg_count - 1 };
                self.frames.push(frame);
                Ok(())
            }
            _ => Err(InterpretError::Runtime("Can only call functions.".to_string())),
        }
    }

    fn capture_upvalue(&mut self, location: usize) -> u64 {
        for &upvalue_id in self.open_upvalues.iter().rev() {
            if let Some(HeapObject::Upvalue(Upvalue::Open(loc))) = self.heap.get(upvalue_id) {
                if *loc == location { return upvalue_id; }
            }
        }
        let new_upvalue_id = self.heap.register(HeapObject::Upvalue(Upvalue::Open(location)));
        self.open_upvalues.push(new_upvalue_id);
        new_upvalue_id
    }
    
    fn close_upvalues(&mut self, last_slot: usize) {
        let mut i = 0;
        while i < self.open_upvalues.len() {
            let up_id = self.open_upvalues[i];
            if let Some(HeapObject::Upvalue(Upvalue::Open(loc))) = self.heap.get_mut(up_id) {
                if *loc >= last_slot {
                    let closed_val = self.stack[*loc];
                    *self.heap.get_mut(up_id).unwrap() = HeapObject::Upvalue(Upvalue::Closed(closed_val));
                    self.open_upvalues.remove(i);
                    continue;
                }
            }
            i += 1;
        }
    }

    fn read_constant(&mut self) -> Result<f64, InterpretError> {
        let const_idx = self.read_byte() as usize;
        let frame = self.frames.last().unwrap();
        let closure = match self.heap.get(frame.closure_id).unwrap() { HeapObject::Closure(c) => c, _ => unreachable!() };
        let func = match self.heap.get(closure.func_id).unwrap() { HeapObject::Function(f) => f, _ => unreachable!() };
        Ok(func.chunk.constants[const_idx])
    }

    fn read_byte(&mut self) -> u8 {
        let frame = self.frames.last_mut().unwrap();
        frame.ip += 1;
        let closure = match self.heap.get(frame.closure_id).unwrap() {
            HeapObject::Closure(c) => c,
            _ => panic!("Invalid state in read_byte: expected closure"),
        };
        let func = match self.heap.get(closure.func_id).unwrap() {
            HeapObject::Function(f) => f,
            _ => panic!("Invalid state in read_byte: expected function"),
        };
        func.chunk.code[frame.ip - 1]
    }

    fn read_short(&mut self) -> usize {
        let frame = self.frames.last_mut().unwrap();
        frame.ip += 2;
        let closure = match self.heap.get(frame.closure_id).unwrap() {
            HeapObject::Closure(c) => c,
            _ => panic!("Invalid state in read_short: expected closure"),
        };
        let func = match self.heap.get(closure.func_id).unwrap() {
            HeapObject::Function(f) => f,
            _ => panic!("Invalid state in read_short: expected function"),
        };
        let high = func.chunk.code[frame.ip - 2] as usize;
        let low = func.chunk.code[frame.ip - 1] as usize;
        (high << 8) | low
    }

    pub fn pop_stack(&mut self) -> Result<f64, InterpretError> {
        self.stack.pop().ok_or_else(|| InterpretError::Runtime("Stack underflow.".to_string()))
    }
}