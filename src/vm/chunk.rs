// src/vm/chunk.rs

use crate::vm::opcode::OpCode;

// A chunk of bytecode representing a compiled script or function.
#[derive(Debug, Default, Clone)]
pub struct Chunk {
    // The sequence of bytecode instructions.
    pub code: Vec<u8>,
    // The pool of constant values (f64 literals) used by the code.
    pub constants: Vec<f64>,
    // A parallel array to code, mapping each byte to a source line number.
    pub lines: Vec<usize>,
    pub names: Vec<String>, 
}

impl Chunk {
    // Creates a new, empty chunk.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_name(&mut self, name: String) -> usize {
        // Here we could also check for existing names to avoid duplicates.
        self.names.push(name);
        self.names.len() - 1
    }
    
    // Appends a byte to the chunk, which can be an OpCode or an operand.
    // Associates the byte with a given source line number for error reporting.
    pub fn write(&mut self, byte: u8, line: usize) {
        self.code.push(byte);
        self.lines.push(line);
    }

    // A convenience method to write an OpCode.
    pub fn write_opcode(&mut self, op: OpCode, line: usize) {
        self.write(op as u8, line);
    }

    // Adds a constant value to the chunk's constant pool.
    // Returns the index of that constant in the pool.
    // The index is used as the operand for OpConstant.
    pub fn add_constant(&mut self, value: f64) -> usize {
        self.constants.push(value);
        self.constants.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::opcode::OpCode;

    #[test]
    fn test_write_and_add_constant() {
        let mut chunk = Chunk::new();
        
        // Add a constant and write the instruction to load it
        let const_idx = chunk.add_constant(1.23);
        chunk.write_opcode(OpCode::OpConstant, 1);
        chunk.write(const_idx as u8, 1);
        
        // Add another instruction
        chunk.write_opcode(OpCode::OpNegate, 2);

        // Verify the contents
        assert_eq!(chunk.code, vec![OpCode::OpConstant as u8, 0, OpCode::OpNegate as u8]);
        assert_eq!(chunk.constants, vec![1.23]);
        assert_eq!(chunk.lines, vec![1, 1, 2]);
    }
}