use crate::vm::chunk::Chunk;

#[derive(Debug, Clone, Default)]
pub struct Function {
    pub arity: usize, // Number of parameters the function expects.
    pub chunk: Chunk,
    pub name: String, // For debugging and error messages.
}

impl Function {
    pub fn new() -> Self {
        Self {
            arity: 0,
            chunk: Chunk::new(),
            name: String::new(),
        }
    }
}