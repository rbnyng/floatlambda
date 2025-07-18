use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Upvalue {
    Open(usize),
    Closed(f64),
}

#[derive(Debug, Clone)]
pub struct Closure {
    pub func_id: u64,
    pub upvalues: Rc<Vec<Upvalue>>,
}