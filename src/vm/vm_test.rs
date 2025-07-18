

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

    #[test]
    fn test_vm_with_globals() {
        let source = "let x = 10 in let y = 20 in (+ x y)";
        assert_eq!(run_vm(source), 30.0);
    }

    #[test]
    fn test_vm_conditionals() {
        assert_eq!(run_vm("if 1 then 10 else 20"), 10.0);
        assert_eq!(run_vm("if 0 then 10 else 20"), 20.0);
        assert_eq!(run_vm("if nil then 10 else 20"), 20.0);
        
        let source = "let x = 5 in if (> x 0) then 99 else -1";
        assert_eq!(run_vm(source), 99.0);
    }
}