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
    Pair(f64, f64),
    Tensor(DifferentiableTensor), 
    Closure(VMClosure), 
    Upvalue(Upvalue), 
    Free(u64),
}

enum HeapObjectTraceData {
    UserFuncEnv(Vec<f64>),
    Pair(f64, f64),
    BuiltinArgs(Vec<f64>),
    TensorParents(Vec<u64>),
    FunctionConstants(Vec<f64>),
    ClosureRefs {
        func_id: u64,
        upvalues_ids: Rc<Vec<u64>>, // Since upvalues is already Rc, we clone the Rc.
    },
    UpvalueValue(f64), // Holds the f64 value if it's a closed upvalue
    None, // For Free and other objects that don't need tracing
}

// --- GC State Management ---

#[derive(Debug, Clone, Copy, PartialEq)]
enum GcColor { White, Gray, Black }

#[derive(Debug, Clone, Copy, PartialEq)]
enum GcState { Idle, Marking, Sweeping }

// --- The Heap with Incremental GC ---

pub struct Heap {
    objects: Vec<Option<HeapObject>>,
    free_list_head: Option<u64>,
    
    // -- Incremental GC State --
    state: GcState,
    marked_bits: Vec<GcColor>,
    worklist: Vec<u64>,
    sweep_pos: usize,
    // A heuristic to trigger GC.
    allocations_since_gc: usize,
}

impl Heap {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            free_list_head: None,
            state: GcState::Idle,
            marked_bits: Vec::new(),
            worklist: Vec::new(),
            sweep_pos: 0,
            allocations_since_gc: 0,
        }
    }

    /// The main allocation function. This is the primary trigger for the GC.
    pub fn register(&mut self, obj: HeapObject) -> u64 {
        self.allocations_since_gc += 1;
        // Trigger a GC cycle if we've allocated a lot, or if we are out of memory.
        if self.allocations_since_gc > 100 || self.free_list_head.is_none() {
            // Perform some GC work. We can tune the amount of work done per step.
            // For simplicity, let's do a few steps.
            for _ in 0..20 {
                if self.state != GcState::Idle {
                    self.gc_step();
                } else {
                    break;
                }
            }
            self.allocations_since_gc = 0;
        }

        if let Some(free_index) = self.free_list_head {
            let next_free_opt = self.objects[free_index as usize].take();
            self.free_list_head = if let Some(HeapObject::Free(next_id)) = next_free_opt {
                if next_id == u64::MAX { None } else { Some(next_id) }
            } else { None };
            
            self.objects[free_index as usize] = Some(obj);
            self.marked_bits[free_index as usize] = GcColor::White; // New objects start white
            free_index
        } else {
            // No free slots, grow the heap.
            self.objects.push(Some(obj));
            self.marked_bits.push(GcColor::White);
            (self.objects.len() - 1) as u64
        }
    }
    
    /// Kicks off a new garbage collection cycle.
    /// The VM calls this with its current set of roots (stack, globals, etc.).
    pub fn start_gc_cycle(&mut self, roots: &[f64]) {
        if self.state != GcState::Idle {
            return; // Already collecting
        }

        self.state = GcState::Marking;
        self.worklist.clear();
        
        for &val in roots {
            if let Some(id) = decode_heap_pointer(val) {
                if self.marked_bits.get(id as usize) == Some(&GcColor::White) {
                    self.marked_bits[id as usize] = GcColor::Gray;
                    self.worklist.push(id);
                }
            }
        }
    }
    
    /// Performs one unit of work for the garbage collector.
    /// This should be called periodically by the VM or allocator.
    pub fn gc_step(&mut self) {
        match self.state {
            GcState::Idle => { /* Do nothing */ },
            GcState::Marking => {
                // Perform one step of marking.
                if let Some(id) = self.worklist.pop() {
                    self.trace_references(id);
                    self.marked_bits[id as usize] = GcColor::Black;
                } else {
                    // Worklist is empty, transition to sweeping phase.
                    self.state = GcState::Sweeping;
                    self.sweep_pos = 0;
                }
            },
            GcState::Sweeping => {
                // Perform one step of sweeping.
                const SWEEP_CHUNK_SIZE: usize = 50;
                let end = (self.sweep_pos + SWEEP_CHUNK_SIZE).min(self.objects.len());

                for i in self.sweep_pos..end {
                    if self.marked_bits[i] == GcColor::White {
                        // This is garbage, add it to the free list.
                        if self.objects[i].is_some() && !matches!(self.objects[i], Some(HeapObject::Free(_))) {
                            let next_free = self.free_list_head;
                            self.objects[i] = Some(HeapObject::Free(next_free.unwrap_or(u64::MAX)));
                            self.free_list_head = Some(i as u64);
                        }
                    } else {
                        // This object is alive, reset its color for the next cycle.
                        self.marked_bits[i] = GcColor::White;
                    }
                }

                if end == self.objects.len() {
                    // Finished sweeping, transition back to idle.
                    self.state = GcState::Idle;
                    self.sweep_pos = 0;
                } else {
                    self.sweep_pos = end;
                }
            }
        }
    }

    /// The original collect is now just a way to force a full, blocking GC cycle.
    /// Useful for tests or when exiting the program.
    pub fn collect_full(&mut self, roots: &[f64]) {
        self.start_gc_cycle(roots);
        while self.state != GcState::Idle {
            // Keep stepping until the entire cycle is complete.
            // A smarter implementation would not have a bounded loop here.
            match self.state {
                GcState::Idle => break,
                GcState::Marking => {
                    while !self.worklist.is_empty() { self.gc_step(); }
                    self.gc_step(); // Final step to transition from Marking to Sweeping
                },
                GcState::Sweeping => {
                    while self.state == GcState::Sweeping { self.gc_step(); }
                }
            }
        }
    }
    
    /// Helper to trace the children of a single object.
    fn trace_references(&mut self, id: u64) {
        // Store relevant data out of the heap object and then process
        let obj_clone_data = {
            if let Some(obj) = self.get(id) {
                match obj {
                    HeapObject::UserFunc(closure) => {
                        // Clone the environment values.
                        // Note: Rc::clone only clones the Rc, not the HashMap itself.
                        HeapObjectTraceData::UserFuncEnv(closure.env.values().copied().collect())
                    }
                    HeapObject::Pair(car, cdr) => HeapObjectTraceData::Pair(*car, *cdr),
                    HeapObject::BuiltinFunc(closure) => HeapObjectTraceData::BuiltinArgs(closure.args.clone()),
                    HeapObject::Tensor(tensor) => {
                        HeapObjectTraceData::TensorParents(tensor.context.as_ref().map_or(vec![], |ctx| ctx.parents.clone()))
                    }
                    HeapObject::Function(func) => HeapObjectTraceData::FunctionConstants(func.chunk.constants.clone()),
                    HeapObject::Closure(closure) => HeapObjectTraceData::ClosureRefs {
                        func_id: closure.func_id,
                        upvalues_ids: closure.upvalues.clone(), // Rc::clone
                    },
                    HeapObject::Upvalue(upvalue) => HeapObjectTraceData::UpvalueValue(if let Upvalue::Closed(val) = upvalue { *val } else { f64::NAN /* placeholder */ }),
                    HeapObject::Free(_) => HeapObjectTraceData::None,
                }
            } else {
                return; // Object not found (e.g., already freed or invalid ID)
            }
        };

        match obj_clone_data {
            HeapObjectTraceData::UserFuncEnv(env_values) => {
                for val in env_values { self.mark_value(val); }
            }
            HeapObjectTraceData::Pair(car, cdr) => {
                self.mark_value(car);
                self.mark_value(cdr);
            }
            HeapObjectTraceData::BuiltinArgs(args) => {
                for arg in args { self.mark_value(arg); }
            }
            HeapObjectTraceData::TensorParents(parent_ids) => {
                for parent_id in parent_ids { self.mark_id(parent_id); }
            }
            HeapObjectTraceData::FunctionConstants(constants) => {
                for constant in constants { self.mark_value(constant); }
            }
            HeapObjectTraceData::ClosureRefs { func_id, upvalues_ids } => {
                self.mark_id(func_id);
                for upvalue_id in upvalues_ids.iter() { self.mark_id(*upvalue_id); }
            }
            HeapObjectTraceData::UpvalueValue(val) => {
                if !val.is_nan() { // Only mark if it was a closed upvalue
                    self.mark_value(val);
                }
            }
            HeapObjectTraceData::None => {}
        }
    }

    /// Marks a value if it's a white pointer.
    fn mark_value(&mut self, val: f64) {
        if let Some(id) = decode_heap_pointer(val) {
            self.mark_id(id);
        }
    }
    
    /// Marks an object by its ID if it's white.
    fn mark_id(&mut self, id: u64) {
        if id as usize >= self.marked_bits.len() || self.marked_bits[id as usize] != GcColor::White {
            return;
        }
        self.marked_bits[id as usize] = GcColor::Gray;
        self.worklist.push(id);
    }

    // The old collect method is renamed to collect_full.
    pub fn collect(&mut self, roots: &[f64]) {
        self.collect_full(roots);
    }
    
    pub fn get(&self, id: u64) -> Option<&HeapObject> {
        self.objects.get(id as usize).and_then(|f| f.as_ref())
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut HeapObject> {
        self.objects.get_mut(id as usize).and_then(|f| f.as_mut())
    }

    pub fn get_tensor_mut(&mut self, id: u64) -> Result<&mut DifferentiableTensor, EvalError> {
        self.get_mut(id)
            .and_then(|obj| match obj {
                HeapObject::Tensor(t) => Some(t),
                _ => None,
            })
            .ok_or_else(|| EvalError::TypeError(format!("Expected a tensor, but heap ID {} is not a tensor.", id)))
    }
    
    pub fn alive_count(&self) -> usize {
        self.objects
            .iter()
            .filter(|o| o.is_some() && !matches!(o, Some(HeapObject::Free(_))))
            .count()
    }

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