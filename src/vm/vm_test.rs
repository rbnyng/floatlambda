

#[cfg(test)]
mod vm_tests {
    use crate::parser::parse;
    use crate::vm;

    fn run_vm(source: &str) -> f64 {
        let term = parse(source).unwrap();
        let chunk = vm::compile(&term).unwrap();
        vm::interpret(&chunk).unwrap()
    }

    #[test]
    fn test_vm_execution() {
        assert_eq!(run_vm("1.0"), 1.0);
        assert_eq!(run_vm("(- (+ 1 2) 4)"), -1.0); // (3 - 4)
        assert_eq!(run_vm("(/ 10 (* 2 2.5))"), 2.0); // 10 / 5
    }
}