// src/vm/opcode.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OpCode {
    // --- Constants and Literals ---
    // Pushes a constant from the chunk's constant pool onto the stack.
    // The operand is a single byte representing the index in the pool.
    OpConstant,

    // Pushes nil (-inf) onto the stack.
    OpNil,
    // Pushes 1.0 (true) onto the stack.
    OpTrue,
    // Pushes 0.0 (false) onto the stack.
    OpFalse,
    // Pops the top value from the stack.
    OpPop,

    // --- Data Structures ---
    OpCons,
    OpCar,
    OpCdr,

    // --- Unary Operations ---
    // Negates the top value on the stack.
    OpNegate,
    // Performs logical NOT on the top value (if value != 0.0 then 0.0 else 1.0).
    OpNot,

    // --- Binary Operations ---
    OpAdd,
    OpSubtract,
    OpMultiply,
    OpDivide,
    OpDivInt, // For integer-like division (floor)
    OpRem,    // For remainder
    
    // --- Comparison ---
    // Strict equality (eq?). Pops two values, pushes 1.0 if equal, 0.0 otherwise.
    OpEqual,
    OpGreater,
    OpLess,
    OpFuzzyEqual,

    // --- Upvalue Opcodes ---
    OpGetUpvalue,
    OpSetUpvalue,
    OpCloseUpvalue,

    // --- Variables ---
    // Defines a new global variable. Operand is an index into the chunk's name pool.
    OpDefineGlobal,
    // Pushes the value of a global variable onto the stack.
    OpGetGlobal,
    // Sets the value of an existing global variable.
    OpSetGlobal,
    // --- Local Variable Opcodes ---
    OpGetLocal,
    OpSetLocal,

    // --- Functions ---
    OpCall,
    OpClosure, 
    OpTailCall, 

    // --- Jumps ---
    // Unconditionally jumps forward by a 16-bit offset.
    OpJump,
    // Jumps forward by a 16-bit offset if the top of the stack is falsey (0.0 or nil).
    OpJumpIfFalse,
    
    // --- Fuzzy Logic Opcode ---
    OpBlend,

    // --- Native Interface ---
    OpNative,

    // --- Control Flow ---
    // Marks the end of a function's execution.
    OpReturn,
}

// Helper to convert a u8 back into an OpCode.
// This is useful in the VM's dispatch loop.
impl From<u8> for OpCode {
    fn from(byte: u8) -> Self {
        // This is safe as long as the byte is a valid OpCode.
        // The VM should ensure it doesn't read invalid bytes.
        unsafe { std::mem::transmute(byte) }
    }
}