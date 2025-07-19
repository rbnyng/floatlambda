#[cfg(test)]
mod vm_tests {
    use crate::{memory::Heap, vm::{self}};

    // Helper to test for a successful run
    fn test_source(source: &str, expected: f64) {
        let mut heap = Heap::new();
        let result = vm::interpret(source, &mut heap).unwrap();
    
        // Check for NaN as a special case
        if expected.is_nan() {
            assert!(result.is_nan(), "Expected NaN, but got {}", result);
        } else {
            // Use the comparison for all other numbers
            assert!((result - expected).abs() < 1e-9, "Expected {}, got {}", expected, result);
        }
    }

    // Helper to test for a runtime error
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
    fn test_vm_recursion() {
        let source = "
            let rec factorial = (λn. 
                if (< n 2) then 1 
                else (* n (factorial (- n 1)))
            ) in (factorial 5)";
        test_source(source, 120.0);
    }

    #[test]
    fn test_vm_tail_call_optimization() {
        // This countdown function will exhaust the call frame stack without TCO.
        // With TCO, it runs in a constant number of frames.
        let source = "
            let rec countdown = (λn.
                if (< n 1) then 42
                else (countdown (- n 1))
            ) in (countdown 10000)
        ";
        test_source(source, 42.0);
    }

    #[test]
    fn test_vm_data_structures() {
        test_source("(car (cons 10 20))", 10.0);
        test_source("(cdr (cons 10 20))", 20.0);
        test_source("let l = (cons 1 (cons 2 nil)) in (car (cdr l))", 2.0);
    }

    #[test]
    fn test_vm_data_structure_errors() {
        test_runtime_error("(car 123)");
        test_runtime_error("(cdr 42.0)");
        test_runtime_error("(car nil)");
    }
    
    #[test]
    fn test_vm_native_functions() {
        test_source("(sqrt 16)", 4.0);
        test_source("(sqrt -1)", f64::NAN);
        test_source("pi", std::f64::consts::PI);
    }

    #[test]
    fn test_vm_native_print() {
        // Can't easily assert stdout, but we can ensure it runs without error
        // and returns the correct success value (1.0).
        // "Hi" -> (cons 72 (cons 105 nil))
        let source = "let s = (cons 72.0 (cons 105.0 nil)) in (print s)";
        test_source(source, 1.0);
    }
}