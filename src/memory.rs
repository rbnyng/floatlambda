// src/memory.rs

use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::Term;
use crate::ml::tensor::DifferentiableTensor;
use crate::error::EvalError;
use crate::vm::function::Function;
use crate::vm::closure::{Closure as VMClosure, Upvalue};

// --- Core Data Structures ---
pub const NIL_VALUE: f64 = f64::NEG_INFINITY;

pub type Environment = Rc<HashMap<String, f64>>;

#[derive(Debug, Clone)]
pub struct ASTClosure {
    pub param: String,
    pub body: Box<Term>,
    pub env: Environment,
}

#[derive(Debug, Clone)]
pub struct BuiltinClosure {
    pub op: String,
    pub arity: usize,
    pub args: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum HeapObject {
    Function(Function), 
    UserFunc(ASTClosure),
    BuiltinFunc(BuiltinClosure),
    Pair(f64, f64), // The classic "cons cell" for building lists.
    Tensor(DifferentiableTensor), 
    Closure(VMClosure), 
    Upvalue(Upvalue), 
    Free(u64), // Points to the next free slot
}

// A central table to store all living heap-allocated objects.
pub struct Heap {
    objects: Vec<Option<HeapObject>>,
    free_list_head: Option<u64>,
}

impl Heap {
    // A helper for debugging to find the ID of a given object. Inefficient because it's O(N) but whatever; it's a debug tool
    pub fn find_id(&self, obj_to_find: &HeapObject) -> Option<u64> {
        let ptr_to_find = obj_to_find as *const HeapObject;
        for (i, obj_opt) in self.objects.iter().enumerate() {
            if let Some(obj) = obj_opt {
                if (obj as *const HeapObject) == ptr_to_find {
                    return Some(i as u64);
                }
            }
        }
        None
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut HeapObject> {
        self.objects.get_mut(id as usize).and_then(|f| f.as_mut())
    }

    // Helper to get a mutable tensor, returning a proper error
    pub fn get_tensor_mut(&mut self, id: u64) -> Result<&mut DifferentiableTensor, EvalError> {
        self.get_mut(id)
            .and_then(|obj| match obj {
                HeapObject::Tensor(t) => Some(t),
                _ => None,
            })
            .ok_or_else(|| EvalError::TypeError(format!("Expected a tensor, but heap ID {} is not a tensor.", id)))
    }

    pub fn new() -> Self {
        Self { objects: Vec::new(), free_list_head: None }
    }

    pub fn register(&mut self, obj: HeapObject) -> u64 {
        if let Some(free_index) = self.free_list_head {
            // Pop the head of the free list
            let next_free = self.objects[free_index as usize].take();
            self.free_list_head = if let Some(HeapObject::Free(next_id)) = next_free {
                if next_id == u64::MAX { None } else { Some(next_id) } // Handle sentinel
            } else {
                None
            };            
            // Place the new object in the reclaimed slot
            self.objects[free_index as usize] = Some(obj);
            return free_index;
        }

        // If the free list is empty, fall back to the old method
        self.objects.push(Some(obj));
        (self.objects.len() - 1) as u64
    }

    pub fn get(&self, id: u64) -> Option<&HeapObject> {
        self.objects.get(id as usize).and_then(|f| f.as_ref())
    }

    // The garbage collector
    pub fn collect(&mut self, roots: &[f64]) {
        let mut marked = vec![false; self.objects.len()];
        let mut worklist: Vec<u64> = roots
            .iter()
            .filter_map(|val| decode_heap_pointer(*val))
            .collect();
        
        while let Some(id) = worklist.pop() {
            if id as usize >= marked.len() || marked[id as usize] {
                continue;
            }
            marked[id as usize] = true;

            // The GC must trace through all heap object types.
            if let Some(obj) = self.get(id) {
                match obj {
                    HeapObject::UserFunc(closure) => {
                        // A ASTClosure is a root for objects in its environment.
                        for val in closure.env.values() {
                            if let Some(child_id) = decode_heap_pointer(*val) {
                                worklist.push(child_id);
                            }
                        }
                    }
                    HeapObject::Pair(car, cdr) => {
                        // A pair is a root for the objects in its car and cdr.
                        if let Some(car_id) = decode_heap_pointer(*car) {
                            worklist.push(car_id);
                        }
                        if let Some(cdr_id) = decode_heap_pointer(*cdr) {
                            worklist.push(cdr_id);
                        }
                    }
                    HeapObject::BuiltinFunc(closure) => { 
                        // Trace through partially applied builtins
                        for arg in &closure.args {
                            if let Some(child_id) = decode_heap_pointer(*arg) {
                                worklist.push(child_id);
                            }
                        }
                    }
                    HeapObject::Tensor(tensor) => {
                        if let Some(ctx) = &tensor.context {
                            for &parent_id in &ctx.parents {
                                worklist.push(parent_id);
                            }
                        }
                    }
                    HeapObject::Free(_) => {
                        // Free slots contain no live data
                    }
                    HeapObject::Function(func) => {
                        // A function's constants can be heap pointers (e.g., to other functions
                        // for creating closures). We trace them.
                        for constant in &func.chunk.constants {
                            if let Some(const_id) = decode_heap_pointer(*constant) {
                                worklist.push(const_id);
                            }
                        }
                    }
                    HeapObject::Closure(closure) => {
                        // A closure holds a reference to its function definition and its upvalues.
                        // We trace both to keep them alive.
                        worklist.push(closure.func_id);
                        for &upvalue_id in closure.upvalues.iter() {
                            worklist.push(upvalue_id);
                        }
                    }
                    HeapObject::Upvalue(upvalue) => {
                        // An upvalue is either open (pointing to the stack, not the heap)
                        // or closed. If it's closed, it holds a value which could be a heap
                        // pointer, so we trace it.
                        if let Upvalue::Closed(val) = upvalue {
                            if let Some(val_id) = decode_heap_pointer(*val) {
                                worklist.push(val_id);
                            }
                        }
                    }
                }
            }
        }

        // --- Sweep Phase ---
        self.free_list_head = None; // Reset the free list
        for i in (0..self.objects.len()).rev() { // Iterate backwards
            if !marked[i] {
                // This object is garbage, add it to the free list.
                let next_free = self.free_list_head;
                self.objects[i] = Some(HeapObject::Free(next_free.unwrap_or(u64::MAX))); 
                self.free_list_head = Some(i as u64);
            }
        }
    }

    // Helper for debugging in the REPL
    pub fn alive_count(&self) -> usize {
        self.objects
            .iter()
            .filter(|o| match o {
                // Free slots and empty slots are not "alive"
                Some(HeapObject::Free(_)) | None => false,
                // Any other Some(...) variant is alive
                Some(_) => true,
            })
            .count()
    }
}

// --- NaN-Boxing Crazy Town ---

const QNAN: u64 = 0x7ff8000000000000;
const PAYLOAD_MASK: u64 = 0x0000ffffffffffff;

pub fn encode_heap_pointer(id: u64) -> f64 {
    f64::from_bits(QNAN | id)
}

pub fn decode_heap_pointer(val: f64) -> Option<u64> {
    if (val.to_bits() & 0x7ff8000000000000) == QNAN {
        Some(val.to_bits() & PAYLOAD_MASK)
    } else {
        None
    }
}