use crate::vm::chunk::Chunk;

#[derive(Debug, Clone, Default)]
pub struct Function {
    pub arity: usize, // Number of parameters the function expects.
    pub chunk: Chunk,
    pub name: String, // For debugging and error messages.
    pub upvalue_count: usize, 
}

impl Function {
    pub fn new() -> Self {
        Self {
            arity: 0,
            chunk: Chunk::new(),
            name: String::new(),
            upvalue_count: 0, 
        }
    }
}