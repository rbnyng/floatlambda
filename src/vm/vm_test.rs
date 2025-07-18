#[cfg(test)]
mod vm_tests {
    use crate::{memory::Heap, vm};

    // Helper to parse, compile, and run source, then assert the result.
    fn test_source(source: &str, expected: f64) {
        let mut heap = Heap::new();
        let result = vm::interpret(source, &mut heap).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_vm_simple_expressions() {
        test_source("1.0", 1.0);
        test_source("(- (+ 1 2) 4)", -1.0);
        test_source("(/ 10 (* 2 2.5))", 2.0);
    }

    #[test]
    fn test_vm_globals() {
        // Simple definition and retrieval
        test_source("let x = 10 in let y = 20 in (+ x y)", 30.0);
        
        // Use a global to initialize another
        test_source("let x = 10 in let y = (+ x 5) in y", 15.0);
        
        // Shadowing a global variable
        test_source("let x = 1 in let x = 2 in x", 2.0);
    }
    
    #[test]
    fn test_vm_conditionals() {
        // Test basic branches
        test_source("if 1 then 10 else 20", 10.0);
        test_source("if 0 then 10 else 20", 20.0);
        
        // Test other "truthy" values
        test_source("if 42 then 10 else 20", 10.0);
        test_source("if -1 then 10 else 20", 10.0);

        // Test "falsey" value (nil)
        test_source("if nil then 10 else 20", 20.0);
        
        // Test with expressions
        let source = "let x = 5 in if (> x 0) then 99 else -1";
        test_source(source, 99.0);
    }

    #[test]
    fn test_vm_nested_conditionals() {
        test_source("if 1 then (if 0 then 1 else 2) else 3", 2.0);
        test_source("if 0 then 1 else (if 1 then 2 else 3)", 2.0);
        test_source("let x = 10 in if (< x 20) then (if (> x 5) then 100 else 200) else 300", 100.0);
    }

    #[test]
    fn test_if_as_expression() {
        test_source("(+ 100 (if 1 then 1 else 2))", 101.0);
        test_source("(+ 100 (if 0 then 1 else 2))", 102.0);
    }
}