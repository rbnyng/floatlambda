#[cfg(test)]
mod vm_tests {
    use crate::{memory::Heap, vm::{self}};

    // Helper to test for a successful run
    fn test_source(source: &str, expected: f64) {
        let mut heap = Heap::new();
        // The main `interpret` function now orchestrates everything.
        let result = vm::interpret(source, &mut heap).unwrap();
        assert!((result - expected).abs() < 1e-9); // Use tolerance for float comparison
    }

    // New helper to test for a runtime error
    fn test_runtime_error(source: &str) {
        let mut heap = Heap::new();
        let result = vm::interpret(source, &mut heap);
        assert!(matches!(result, Err(vm::vm::InterpretError::Runtime(_))));
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

    #[test]
    fn test_vm_simple_function_call() {
        let source = "let add10 = (λx. (+ x 10)) in (add10 5)";
        test_source(source, 15.0);
    }

    #[test]
    fn test_vm_function_behavior() {
        // Test a multi-argument function
        test_source("let add = (λx. λy. (+ x y)) in ((add 10) 20)", 30.0);

        // Test that a local parameter shadows a global variable
        test_source("let a = 100 in let f = (λa. (+ a 1)) in (f 10)", 11.0);

        // Test that functions can be passed as values
        test_source("let f = (λx. (+ x 1)) in let g = f in (g 5)", 6.0);

        // Test an `if` expression that returns a function
        test_source("let f = (if 1 then (λx.(+ x 1)) else (λx.(* x 2))) in (f 10)", 11.0);
        test_source("let f = (if 0 then (λx.(+ x 1)) else (λx.(* x 2))) in (f 10)", 20.0);
    }
    
    #[test]
    fn test_vm_arity_errors() {
        // Too few arguments - should now pass
        test_runtime_error("let f = (λx. λy. x) in (f 1)");

        // Too many arguments - uncomment this, it should now pass
        test_runtime_error("let f = (λx. x) in (f 1 2)");
    }

    #[test]
    fn test_vm_recursion() {
        let source = "
            let rec factorial = (λn. 
                if (< n 2) then 1 
                else (* n (factorial (- n 1)))
            ) in (factorial 5)";
        test_source(source, 120.0);
    }
}