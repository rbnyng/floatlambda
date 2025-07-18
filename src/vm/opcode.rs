// src/vm/opcode.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OpCode {
    // --- Constants and Literals ---
    /// Pushes a constant from the chunk's constant pool onto the stack.
    /// The operand is a single byte representing the index in the pool.
    OpConstant,

    /// Pushes `nil` (-inf) onto the stack.
    OpNil,
    /// Pushes `1.0` (true) onto the stack.
    OpTrue,
    /// Pushes `0.0` (false) onto the stack.
    OpFalse,

    // --- Unary Operations ---
    /// Negates the top value on the stack.
    OpNegate,
    /// Performs logical NOT on the top value (if value != 0.0 then 0.0 else 1.0).
    OpNot,

    // --- Binary Operations ---
    OpAdd,
    OpSubtract,
    OpMultiply,
    OpDivide,

    // --- Comparison ---
    /// Strict equality (eq?). Pops two values, pushes 1.0 if equal, 0.0 otherwise.
    OpEqual,
    OpGreater,
    OpLess,

    // --- Control Flow ---
    /// Marks the end of a function's execution.
    OpReturn,
}

// Helper to convert a u8 back into an OpCode.
// This will be useful in the VM's dispatch loop.
impl From<u8> for OpCode {
    fn from(byte: u8) -> Self {
        // This is safe as long as the byte is a valid OpCode.
        // The VM should ensure it doesn't read invalid bytes.
        unsafe { std::mem::transmute(byte) }
    }
}