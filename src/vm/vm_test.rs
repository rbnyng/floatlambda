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

    // Helper to test for a successful run with wider tolerance for calculus results
    fn test_calculus_source(source: &str, expected: f64) {
        let mut heap = Heap::new();
        let result = vm::interpret(source, &mut heap).unwrap();
        // Calculus results are approximations, allow wider tolerance (e.g., 0.1)
        assert!((result - expected).abs() < 0.1, "Expected {}, got {}", expected, result);
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
    fn test_vm_strict_equality_eq() {
        // Numbers
        test_source("(eq? 1.0 1.0)", 1.0);
        test_source("(eq? 1.0 1.000000001)", 0.0);
        test_source("(eq? 0.0 -0.0)", 0.0); // Strict bit-wise equality differs

        // Pointers (should be equal only if they are the same object)
        let source = "let mylist = (cons 1 2) in (eq? mylist mylist)";
        test_source(source, 1.0);
        
        let source_neq = "let l1 = (cons 1 2) in let l2 = (cons 1 2) in (eq? l1 l2)";
        test_source(source_neq, 0.0); // Different objects, not equal

        // Nil
        test_source("(eq? nil nil)", 1.0);
        test_source("(eq? nil 0.0)", 0.0);
    }

    #[test]
    fn test_vm_fuzzy_equality_double_equals() {
        // Perfect match
        test_calculus_source("(== 1.0 1.0)", 1.0);

        // Close match (high similarity)
        let source_close = "(== 1000.0 1001.0)";
        let expected_close = (-(1.0f64 / 1001.0)).exp(); // ~0.999
        test_calculus_source(source_close, expected_close);

        // Further match (lower similarity)
        let source_far = "(== 1.0 2.0)";
        let expected_far = (-(1.0f64 / 2.0)).exp(); 
        test_calculus_source(source_far, expected_far);
        
        // Small numbers
        let source_small = "(== 0.1 0.2)";
        let expected_small = (-(0.1f64 / 1.0)).exp();  // scale factor is 1.0, ~0.904
        test_calculus_source(source_small, expected_small);
    }

    #[test]
    fn test_vm_fuzzy_eq_on_pointers_is_nan() {
        // Fuzzy equality on non-numbers (pointers, nil) should produce NaN
        // because the formula involves arithmetic.
        test_source("(== nil nil)", f64::NAN);
        let source = "let mylist = (cons 1 2) in (== mylist mylist)";
        test_source(source, f64::NAN);
    }

    #[test]
    fn test_vm_if_statement() {
        // --- Discrete cases ---
        // True condition
        test_source("if 1.0 then 100 else 200", 100.0);
        // Any non-zero, non-nil value > 1 is clamped to 1.0
        test_source("if 42.0 then 100 else 200", 100.0);
        
        // False condition
        test_source("if 0.0 then 100 else 200", 200.0);
        // Nil is also false
        test_source("if nil then 100 else 200", 200.0);
        // Negative values are clamped to 0.0
        test_source("if -5.0 then 100 else 200", 200.0);

        // --- Fuzzy cases ---
        // 50/50 blend
        test_source("if 0.5 then 100 else 200", 150.0);
        // 70/30 blend
        test_source("if 0.7 then 100 else 200", 130.0); // 0.7*100 + 0.3*200 = 70 + 60
        // 25/75 blend
        test_source("if 0.25 then 10 else 30", 25.0); // 0.25*10 + 0.75*30 = 2.5 + 22.5

        // Test with expressions
        let source = "let x = 0.8 in if x then (+ 10 10) else (* 10 10)"; // 0.8*20 + 0.2*100 = 16 + 20
        test_source(source, 36.0);
    }

    #[test]
    fn test_vm_nested_blending() {
        // if 0.5 then (if 0.5 then 10 else 20) else 30
        // -> if 0.5 then 15 else 30
        // -> 0.5 * 15 + 0.5 * 30 = 7.5 + 15 = 22.5
        test_source("if 0.5 then (if 0.5 then 10 else 20) else 30", 22.5);
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

    #[test]
    fn test_vm_diff_constant_function() {
        let source = "(diff (λx. 5.0) 10.0)";
        test_calculus_source(source, 0.0);
    }

    #[test]
    fn test_vm_diff_identity_function() {
        let source = "(diff (λx. x) 42.0)";
        test_calculus_source(source, 1.0);
    }

    #[test]
    fn test_vm_diff_x_squared() {
        let source = "let f = (λx. (* x x)) in (diff f 5.0)";
        test_calculus_source(source, 10.0); // d/dx(x^2) = 2x, at x=5 is 10
    }

    #[test]
    fn test_vm_diff_x_cubed() {
        let source = "let f = (λx. (* x (* x x))) in (diff f 2.0)";
        test_calculus_source(source, 12.0); // d/dx(x^3) = 3x^2, at x=2 is 12
    }

    #[test]
    fn test_vm_integrate_constant() {
        let source = "(integrate (λx. 1.0) 0.0 5.0)";
        test_calculus_source(source, 5.0); // ∫₀⁵ 1 dx = 5
    }

    #[test]
    fn test_vm_integrate_x() {
        let source = "(integrate (λx. x) 0.0 2.0)";
        test_calculus_source(source, 2.0); // ∫₀² x dx = x²/2 |₀² = 2
    }

    #[test]
    fn test_vm_integrate_x_squared() {
        let source = "let f = (λx. (* x x)) in (integrate f 0.0 3.0)";
        test_calculus_source(source, 9.0); // ∫₀³ x² dx = x³/3 |₀³ = 9
    }

    #[test]
    fn test_vm_higher_order_diff() {
        // Second derivative: f''(x) = d/dx(d/dx(f(x)))
        let source = "
            let f = (λx. (* x (* x x))) in   # f(x) = x³
            let f_prime = (λx. (diff f x)) in # f'(x)
            (diff f_prime 2.0)                 # f''(2)
        ";
        // f(x) = x³, f'(x) = 3x², f''(x) = 6x, f''(2) = 12
        test_calculus_source(source, 12.0);
    }

    #[test]
    fn test_vm_fundamental_theorem_of_calculus() {
        // F(x) = ∫₀ˣ t² dt, then F'(x) should equal x²
        let source = "
            let f = (λx. (* x x)) in
            let F = (λx. (integrate f 0.0 x)) in
            let x_val = 5.0 in
            let derivative_of_F = (diff F x_val) in
            let original_f = (f x_val) in
            ((- derivative_of_F) original_f) # Should be close to 0
        ";
        test_calculus_source(source, 0.0);
    }
}