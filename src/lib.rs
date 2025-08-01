// src/lib.rs

// --- Module Declarations ---
// These lines declare the other files as modules of this library.
pub mod ast;
pub mod error;
pub mod memory;
pub mod parser;
pub mod interpreter;
pub mod ml; 
pub mod math; 
pub mod vm;

// --- Public API Re-exports ---
// This makes the core components available to users of the library
// without them needing to know the internal file structure.
pub use ast::Term;
pub use error::{EvalError, ParseError};
pub use memory::{Heap, NIL_VALUE};
pub use parser::parse;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::rc::Rc;

    // Helper to run evaluation
    fn eval_ok(input: &str) -> f64 {
        let term = parse(input).unwrap();
        let mut heap = Heap::new();
        term.eval(&Rc::new(HashMap::new()), &mut heap).unwrap()
    }

    fn eval_err(input: &str) -> EvalError {
        let term = parse(input).unwrap();
        let mut heap = Heap::new();
        term.eval(&Rc::new(HashMap::new()), &mut heap).unwrap_err()
    }

    #[test]
    fn test_float_eval() {
        assert_eq!(eval_ok("42.0"), 42.0);
    }

    #[test]
    fn test_var_eval() {
        assert_eq!(eval_ok("let x = 10.0 in x"), 10.0);
        // Test unbound variable error
        assert_eq!(
            eval_err("x"),
            EvalError::UnboundVariable("x".to_string())
        );
    }

    #[test]
    fn test_fuzzy_eq() {
        assert!((interpreter::evaluator::fuzzy_eq(1.0, 1.0) - 1.0).abs() < 0.001);
        assert!(interpreter::evaluator::fuzzy_eq(1.0, 2.0) < 1.0);
        assert!(interpreter::evaluator::fuzzy_eq(1000.0, 1001.0) > 0.99); // Should be very close to 1
    }

    #[test]
    fn test_lambda_application() {
        // (λx.x) 42.0
        assert_eq!(eval_ok("(λx.x 42.0)"), 42.0);
    }

    #[test]
    fn test_binary_operation() {
        // ((+ 1.0) 2.0) should equal 3.0
        assert_eq!(eval_ok("((+ 1.0) 2.0)"), 3.0);
        assert_eq!(eval_ok("((/ 10.0) 2.0)"), 5.0);
        assert_eq!(eval_ok("((/ 10.0) 0.0)"), f64::INFINITY);
    }

    #[test]
    fn test_parse_float() {
        assert_eq!(parse("42.0").unwrap(), Term::Float(42.0));
        assert_eq!(parse("-3.14").unwrap(), Term::Float(-3.14));
        assert_eq!(parse("0.0").unwrap(), Term::Float(0.0));
    }

    #[test]
    fn test_parse_variable() {
        assert_eq!(parse("x").unwrap(), Term::Var("x".to_string()));
        assert_eq!(parse("foo_bar").unwrap(), Term::Var("foo_bar".to_string()));
    }

    #[test]
    fn test_parse_builtin() {
        assert_eq!(parse("+").unwrap(), Term::Builtin("+".to_string()));
        assert_eq!(parse("sqrt").unwrap(), Term::Builtin("sqrt".to_string()));
    }

    #[test]
    fn test_parse_lambda() {
        let lambda = parse("λx.x").unwrap();
        assert_eq!(lambda, Term::Lam("x".to_string(), Box::new(Term::Var("x".to_string()))));
        
        // Test backslash syntax too
        let lambda2 = parse("\\x.x").unwrap();
        assert_eq!(lambda2, lambda);
    }

    #[test]
    fn test_parse_application() {
        let app = parse("(f x)").unwrap();
        assert_eq!(app, Term::App(
            Box::new(Term::Var("f".to_string())),
            Box::new(Term::Var("x".to_string()))
        ));
        
        // Test multiple arguments (parsed as curried applications)
        let app2 = parse("(+ 1.0 2.0)").unwrap();
        assert_eq!(app2, Term::App(
            Box::new(Term::App(
                Box::new(Term::Builtin("+".to_string())),
                Box::new(Term::Float(1.0))
            )),
            Box::new(Term::Float(2.0))
        ));
    }

    #[test]
    fn test_fuzzy_logic() {
        assert_eq!(eval_ok("((fuzzy_and 0.8) 0.6)"), 0.48);
        assert!((eval_ok("((fuzzy_or 0.8) 0.6)") - 0.92).abs() < 0.0001);
        assert_eq!(eval_ok("(fuzzy_not 0.3)"), 0.7);
        assert_eq!(eval_ok("(fuzzy_not 1.5)"), 0.0);
    }

    #[test]
    fn test_blended_conditional() {
        // if 0.7 then 100.0 else 200.0
        // = 0.7 * 100.0 + 0.3 * 200.0 = 70.0 + 60.0 = 130.0
        assert_eq!(eval_ok("if 0.7 then 100.0 else 200.0"), 130.0);
    }

    #[test]
    fn test_parse_if() {
        let if_expr = parse("if 0.5 then 10.0 else 20.0").unwrap();
        assert!(matches!(if_expr, Term::If(_, _, _)));
        assert_eq!(eval_ok("if 0.5 then 10.0 else 20.0"), 15.0); // 0.5 * 10.0 + 0.5 * 20.0
    }

    #[test]
    fn test_truth_decay_concept() {
        // Test that repeated operations can change "true" values
        let mut truth = 1.0;
        for _ in 0..1000 {
            truth = truth * 1.0000001;
        }
        assert!(truth != 1.0);
        assert!(truth > 1.0);
    }
    
    #[test]
    fn test_complex_fuzzy_expression() {
        // Test: if ((fuzzy_and 0.8) 0.6) then 100.0 else 200.0
        // 0.8 * 0.6 = 0.48
        // 0.48 * 100.0 + 0.52 * 200.0 = 48.0 + 104.0 = 152.0
        assert_eq!(eval_ok("if ((fuzzy_and 0.8) 0.6) then 100.0 else 200.0"), 152.0);
    }
    
    #[test]
    fn test_nan_handling() {
        // sqrt(-1) should be NaN
        let result = eval_ok("(sqrt -1.0)");
        assert!(result.is_nan());
    }
    
    #[test]
    fn test_improved_fuzzy_eq() {
        assert_eq!(interpreter::evaluator::fuzzy_eq(1.0, 1.0), 1.0);
        // exp(-1.0 / 2.0) = exp(-0.5)
        assert!((interpreter::evaluator::fuzzy_eq(1.0, 2.0) - 0.60653).abs() < 0.0001);
        // exp(-1.0 / 1001.0)
        assert!((interpreter::evaluator::fuzzy_eq(1000.0, 1001.0) - 0.99900).abs() < 0.0001);
    }

    #[test]
    fn test_robust_parser_keywords() {
        // Test that "iff" is parsed as a variable (not the "if" keyword)
        assert!(matches!(parse("iff").unwrap(), Term::Var(_)));
        
        // Test that malformed if statements are caught
        assert!(parse("if 1 thenn 2 else 3").is_err());
        assert!(parse("if 1 then 2 elsee 3").is_err());
        
        // Test that we can distinguish if from iff
        assert!(matches!(parse("if 1 then 2 else 3").unwrap(), Term::If(_, _, _)));
        
        // Test that we can use "iff" as a variable name in a lambda
        let iff_usage = parse("λiff.iff").unwrap();
        assert!(matches!(iff_usage, Term::Lam(_, _)));
        assert_eq!(eval_ok("((λiff.iff) 42.0)"), 42.0); // And it works!
    }

    // --- Tests for First-Class Functions, Currying, Let ---

    #[test]
    fn test_lone_lambda_is_a_value() {
        let result = eval_ok("λx.x");
        assert!(result.is_nan()); // It's a NaN-boxed value
        assert!(memory::decode_heap_pointer(result).is_some()); // It decodes to a valid ID
    }
    
    #[test]
    fn test_let_binding_simple() {
        assert_eq!(eval_ok("let x = 10 in ((+ x) 5)"), 15.0);
        assert_eq!(eval_ok("let x = 3 in let y = 4 in ((* x) y)"), 12.0);
    }
    
    #[test]
    fn test_currying_builtin() {
        let code = "let add5 = (+ 5) in (add5 10)";
        assert_eq!(eval_ok(code), 15.0);
        let code2 = "let sub_from_10 = (- 10) in (sub_from_10 3)"; // 10 - 3 = 7
        assert_eq!(eval_ok(code2), 7.0);
    }
    
    #[test]
    fn test_higher_order_function_const() {
        let code = "let const = (λx.λy.x) in ((const 42) 100)";
        assert_eq!(eval_ok(code), 42.0);
    }
    
    #[test]
    fn test_higher_order_function_apply() {
        let code = "let apply = (λf.λx.(f x)) in (let add1 = (+ 1) in ((apply add1) 10))";
        assert_eq!(eval_ok(code), 11.0);
    }
    
    #[test]
    fn test_application_of_non_function() {
        assert_eq!(
            eval_err("(10 20)"),
            EvalError::TypeError("Cannot apply a non-function value: 10".to_string())
        );
    }

    #[test]
    fn test_emergent_property_nan_propagation() {
        // This test demonstrates that simple arithmetic on a function might not "break" it
        // because NaN + number often just returns the same NaN.
        let code = "let f = (λx.x) in (let g = (+ f 0) in (g 42))";
        // g is effectively the same function as f. The application should succeed.
        assert_eq!(eval_ok(code), 42.0);

        // Even NaN * 1.0 typically returns the same NaN payload.
        let code2 = "let f = (λx.x) in (let g = ((* f) 1.0) in (g 42))";
        assert_eq!(eval_ok(code2), 42.0);
    }
    
    #[test]
    fn test_if_statement_with_function_as_condition() {
        // The if statement will treat the NaN-boxed function as a number.
        // A NaN value, when clamped by max(0.0).min(1.0), typically becomes 0.0.
        // Therefore, the else branch should be taken.
        let code = "if (λx.x) then 100 else 200";
        assert_eq!(eval_ok(code), 200.0);
    }
    
    #[test]
    fn test_complex_expression_with_let() {
        let code = "let a = 10 in let b = 20 in if ((== a) 10.1) then b else a";
        let result = eval_ok(code);
        // fuzzy_eq(10.0, 10.1) is exp(-0.1 / 10.1) approx 0.99014
        // 0.99014 * 20.0 + (1.0 - 0.99014) * 10.0 = 19.8028 + 0.0986 = 19.9014
        assert!((result - 19.9014).abs() < 0.0001);
    }

    // --- Complex Data Structures ---

    #[test]
    fn test_list_of_lists() {
        let code = "let list = (cons (cons 1 2) (cons (cons 3 4) nil)) in (car (car (cdr list)))";
        // list = ((1 . 2) . ((3 . 4) . nil))
        // (cdr list) -> ((3 . 4) . nil)
        // (car (cdr list)) -> (3 . 4)
        // (car (car (cdr list))) -> 3
        assert_eq!(eval_ok(code), 3.0);
    }

    #[test]
    fn test_non_nil_terminated_list() {
        // A "dotted pair" at the end.
        let code = "let pair = (cons 1 2) in (cdr pair)";
        assert_eq!(eval_ok(code), 2.0);
    }

    #[test]
    fn test_can_cons_anything() {
        // (cons (λx.x) (cons 10 nil))
        let code = "let f = (λx.x) in let l = (cons f nil) in ((car l) 42)";
        assert_eq!(eval_ok(code), 42.0);
    }
    
    // --- Higher-Order Functions ---

    #[test]
    fn test_higher_order_map() {
        let map_code = "let rec map = (λf.λl.if (eq? l nil) then nil else (cons (f (car l)) (map f (cdr l))))";
        let list_code = "let mylist = (cons 10 (cons 20 (cons 30 nil)))";
        let add5 = "(λx.(+ x 5))";
        
        let full_code = format!("let apply = (λf.λx.(f x)) in {} in {} in let mapped_list = ((map {}) mylist) in (car (cdr mapped_list))", map_code, list_code, add5);
        // mapped_list would be (15 . (25 . (35 . nil)))
        // (cdr mapped_list) -> (25 . (35 . nil))
        // (car (cdr mapped_list)) -> 25
        assert_eq!(eval_ok(&full_code), 25.0);
    }
    
    #[test]
    fn test_higher_order_filter() {
        let filter_code = "let rec filter = (λp.λl.
            if (eq? l nil) then nil 
            else (
                let head = (car l) in
                let tail = (cdr l) in
                if (p head) then (cons head (filter p tail)) else (filter p tail)
            ))";
        let list_code = "let mylist = (cons 1 (cons 2 (cons 3 (cons 4 nil))))";
        
        let is_gt2 = "(λx.(> x 2))";
        
        let full_code = format!("{} in {} in let filtered_list = ((filter {}) mylist) in (car (cdr filtered_list))", filter_code, list_code, is_gt2);
        // filtered_list should be (3 . (4 . nil))
        // (cdr filtered_list) -> (4 . nil)
        // (car (cdr filtered_list)) -> 4
        assert_eq!(eval_ok(&full_code), 4.0);
    }
    
    #[test]
    fn test_higher_order_fold_left_aka_reduce() {
        let fold_code = "let rec fold = (λf.λacc.λl.
            if (eq? l nil) then acc 
            else (
                fold f (f acc (car l)) (cdr l)
            ))";
        let list_code = "let mylist = (cons 1 (cons 2 (cons 3 (cons 4 nil))))";
        let sum_op = "(λa.λb.(+ a b))";
        
        // Sum of list: fold(+, 0, list)
        let full_code = format!("{} in {} in (((fold {}) 0) mylist)", fold_code, list_code, sum_op);
        // (1 + 2 + 3 + 4) = 10
        assert_eq!(eval_ok(&full_code), 10.0);
    }

    // --- Recursion and Scope ---

    #[test]
    fn test_closure_captures_correct_env() {
        let code = "let make_adder = (λx. (λy. (+ x y))) in let add5 = (make_adder 5) in (add5 10)";
        assert_eq!(eval_ok(code), 15.0);
        
        // Ensure it doesn't capture a later binding of x
        let code2 = "let make_adder = (λx. (λy. (+ x y))) in let add5 = (make_adder 5) in let x = 100 in (add5 10)";
        assert_eq!(eval_ok(code2), 15.0);
    }
    
    // --- Garbage Collector Stress Tests ---

    #[test]
    fn test_gc_on_long_list() {
        let mut heap = Heap::new();
        let global_env = HashMap::new();
        
        let mut list_str = "nil".to_string();
        for i in 0..100 {
            list_str = format!("(cons {} {})", i, list_str);
        }

        let term = parse(&list_str).unwrap();
        let eval_env = Rc::new(global_env);
        let list_val = term.eval(&eval_env, &mut heap).unwrap();
        
        // Before GC, 100 cons calls = 100 pairs.
        let initial_count = heap.alive_count();
        assert!(initial_count > 100); // It will be more than just the pairs.

        heap.collect_full(&[list_val]);
        assert_eq!(heap.alive_count(), 100); // After GC, only the 100 pairs should remain.

        heap.collect_full(&[]);
        assert_eq!(heap.alive_count(), 0);
    }

    #[test]
    fn test_gc_reclaims_part_of_a_structure() {
        let mut heap = Heap::new();
        let env = Rc::new(HashMap::new());

        // 1. Evaluate the list creation expression directly.
        let list_term = parse("(cons 1 (cons 2 (cons 3 nil)))").unwrap();
        let list_val = list_term.eval(&env, &mut heap).unwrap();

        // 2. Run the GC with the list head as the root. This cleans up temporary objects
        // created during evaluation (like intermediate cons builtin closures).
        heap.collect_full(&[list_val]);
        assert_eq!(heap.alive_count(), 3); // 3 pairs in the list.

        // 3. Create a new environment for the next step.
        let mut next_env_map = HashMap::new();
        next_env_map.insert("list".to_string(), list_val);
        let next_env = Rc::new(next_env_map);

        // 4. Evaluate (cdr list) in the controlled environment.
        let cdr_term = parse("(cdr list)").unwrap();
        let tail_val = cdr_term.eval(&next_env, &mut heap).unwrap();

        // After this eval, we might have a temporary BuiltinFunc for cdr.
        // Let's check alive_count before the final GC. It should be > 3.
        let count_before_final_gc = heap.alive_count();
        assert!(count_before_final_gc >= 3); 
        
        // 5. Now, collect garbage, only keeping the tail rooted.
        heap.collect_full(&[tail_val]);
        
        // The head of the list (cons 1 ...) is no longer rooted and should be collected.
        // The tail (cons 2 (cons 3 nil)) consists of 2 pairs, which should remain.
        assert_eq!(heap.alive_count(), 2);
    }

    // --- Error Condition Tests ---

    #[test]
    fn test_application_error_types() {
        use memory::decode_heap_pointer;
        assert_eq!(eval_err("(1 2)"), EvalError::TypeError("Cannot apply a non-function value: 1".to_string()));
        assert_eq!(eval_err("(nil 2)"), EvalError::TypeError("Cannot apply a non-function value: nil".to_string()));
        
        let list_ptr_code = "(cons 1 2)";
        let result = eval_ok(list_ptr_code);
        let id = decode_heap_pointer(result).unwrap();
        let expected_err = EvalError::TypeError(format!("Cannot apply a non-function value: Pair<{}>", id));
        assert_eq!(eval_err(&format!("({} 3)", list_ptr_code)), expected_err);
    }
    
    #[test]
    fn test_unbound_variable_in_deeper_expression() {
        let code = "(+ 1 (z 10))"; // z is unbound
        assert_eq!(eval_err(code), EvalError::UnboundVariable("z".to_string()));
    }
    
    #[test]
    fn test_let_rec_requires_lambda() {
        let code = "let rec x = 5 in x";
        assert_eq!(eval_err(code), EvalError::TypeError("let rec expression must result in a heap-allocated value (a function or pair)".to_string()));
    }
    
    #[test]
    fn test_car_cdr_type_errors() {
        assert_eq!(eval_err("(car 1.0)"), EvalError::TypeError("'car' expects a pair, but got a number or nil.".to_string()));
        assert_eq!(eval_err("(cdr nil)"), EvalError::TypeError("'cdr' expects a pair, but got a number or nil.".to_string()));
        let code = "(car (λx.x))";
        assert_eq!(eval_err(code), EvalError::TypeError("'car' expects a pair, but got another heap object.".to_string()));
    }

    // --- String Tests ---

    #[test]
    fn test_string_as_list_manipulation() {
        // let s = "Hi" -> (cons 72.0 (cons 105.0 nil))
        let code = "
            let s = (cons 72.0 (cons 105.0 nil)) in
            let first_char = (car s) in
            # Test comments
            let second_char = (car (cdr s)) in
            second_char
        ";
        assert_eq!(eval_ok(code), 105.0);
    
        // Also check that the list is nil-terminated correctly.
        let code_nil_check = "
            let s = (cons 72.0 (cons 105.0 nil)) in
            (eq? (cdr (cdr s)) nil)
        ";
        assert_eq!(eval_ok(code_nil_check), 1.0);
    }

    #[test]
    fn test_strlen_on_string_list() {
        // Define strlen, which finds the length of a list.
        let strlen_code = "let rec strlen = (λl. if (eq? l nil) then 0 else (+ 1 (strlen (cdr l))))";
        
        // Create the string "cat" -> (cons 99 (cons 97 (cons 116 nil)))
        let str_cat = "(cons 99.0 (cons 97.0 (cons 116.0 nil)))";
    
        let full_code = format!("
            {} in 
            (strlen {})
        ", strlen_code, str_cat);
    
        // The length of "cat" is 3.
        assert_eq!(eval_ok(&full_code), 3.0);
    }

    #[test]
    fn test_builtin_length() {
        assert_eq!(eval_ok("(length nil)"), 0.0);
        let code = "(length (cons 10 (cons 20 (cons 30 nil))))";
        assert_eq!(eval_ok(code), 3.0);
        // Test error on non-list
        assert!(eval_err("(length 1)").is_type_error());
    }

    #[test]
    fn test_builtin_map() {
        let code = "let l = (cons 1 (cons 2 (cons 3 nil))) in let add10 = (+ 10) in (map add10 l)";
        // Compare the string representation for lists as we don't have a deep eq primitive
        assert_eq!(eval_ok(&format!("(car {})", code)), 11.0);
        assert_eq!(eval_ok(&format!("(car (cdr {}))", code)), 12.0);
        assert_eq!(eval_ok(&format!("(car (cdr (cdr {})))", code)), 13.0);
    }

    #[test]
    fn test_builtin_filter() {
        let code = "
            let l = (cons 1 (cons 2 (cons 3 (cons 4 nil)))) in 
            let is_even = (λx. (eq? 0 (rem x 2))) in
            (filter is_even l)
        ";
         assert_eq!(eval_ok(&format!("(car {})", code)), 2.0);
         assert_eq!(eval_ok(&format!("(car (cdr {}))", code)), 4.0);
         assert_eq!(eval_ok(&format!("(length {})", code)), 2.0);
    }

    #[test]
    fn test_builtin_foldl() {
        let code = "
            let l = (cons 1 (cons 2 (cons 3 (cons 4 nil)))) in
            (foldl + 0 l)
        ";
        // 0+1=1, 1+2=3, 3+3=6, 6+4=10
        assert_eq!(eval_ok(code), 10.0);
    }
    
    // Helper for tests to check error variants easily
    impl EvalError {
        fn is_type_error(&self) -> bool { matches!(self, EvalError::TypeError(_)) }
    }

    #[test]
    fn test_basic_derivatives() {
        // Derivative of constant function should be 0
        let code = "(diff (λx.5.0) 10.0)";
        let result = eval_ok(code);
        assert!(result.abs() < 0.001, "d/dx(5) should be ~0, got {}", result);
        
        // Derivative of identity function should be 1
        let code = "(diff (λx.x) 42.0)";
        let result = eval_ok(code);
        assert!((result - 1.0).abs() < 0.001, "d/dx(x) should be ~1, got {}", result);
        
        // Derivative of x² at x=5 should be 10
        let code = "(diff (λx.((* x) x)) 5.0)";
        let result = eval_ok(code);
        assert!((result - 10.0).abs() < 0.01, "d/dx(x²) at x=5 should be ~10, got {}", result);
        
        // Derivative of x³ at x=2 should be 12
        let code = "(diff (λx.((* x) ((* x) x))) 2.0)";
        let result = eval_ok(code);
        assert!((result - 12.0).abs() < 0.01, "d/dx(x³) at x=2 should be ~12, got {}", result);
    }
    
    #[test]
    fn test_basic_integrals() {
        // Integral of constant 1 from 0 to 5 should be 5
        let code = "(integrate (λx.1.0) 0.0 5.0)";
        let result = eval_ok(code);
        assert!((result - 5.0).abs() < 0.01, "∫₀⁵ 1 dx should be ~5, got {}", result);
        
        // Integral of x from 0 to 2 should be 2 (x²/2 from 0 to 2 = 4/2 - 0 = 2)
        let code = "(integrate (λx.x) 0.0 2.0)";
        let result = eval_ok(code);
        assert!((result - 2.0).abs() < 0.01, "∫₀² x dx should be ~2, got {}", result);
        
        // Integral of x² from 0 to 3 should be 9 (x³/3 from 0 to 3 = 27/3 - 0 = 9)
        let code = "(integrate (λx.((* x) x)) 0.0 3.0)";
        let result = eval_ok(code);
        assert!((result - 9.0).abs() < 0.1, "∫₀³ x² dx should be ~9, got {}", result);
    }
    
    #[test]
    fn test_fundamental_theorem_of_calculus() {
        // F(x) = ∫₀ˣ t² dt, then F'(x) should equal x²
        let code = "
            let f = (λx.((* x) x)) in
            let F = (λx.(integrate f 0.0 x)) in
            let x = 3.0 in
            let derivative_of_F = (diff F x) in
            let original_f = (f x) in
            ((- derivative_of_F) original_f)
        ";
        let difference = eval_ok(code);
        assert!(difference.abs() < 0.1, "Fundamental theorem violation: |F'(x) - f(x)| = {}", difference);
    }
    
    #[test]
    fn test_simple_curried_integration() {
        // Test with a builtin function first
        let code = "
            let integrator = (integrate (+ 0) 0.0) in  # f(x) = x
            (integrator 2.0)
        ";
        let result = eval_ok(code);
        println!("Simple curried result: {}", result);
    }
    
    #[test]
    fn test_curried_integration() {
        // Test the 2-argument curried integrate
        let code = "
            let f = (λx.((* x) x)) in
            let integrator = (integrate f 0.0) in
            (integrator 3.0)
        ";
        let result = eval_ok(code);
        assert!((result - 9.0).abs() < 0.1, "Curried integration should give ~9, got {}", result);
        
        // Test partial application usefulness
        let code = "
            let integrate_x_squared_from_zero = (integrate (λx.((* x) x)) 0.0) in
            let area1 = (integrate_x_squared_from_zero 2.0) in
            let area2 = (integrate_x_squared_from_zero 3.0) in
            ((- area2) area1)
        ";
        let result = eval_ok(code);
        // area2 = 9, area1 = 8/3 ≈ 2.67, difference ≈ 6.33
        assert!((result - (9.0 - 8.0/3.0)).abs() < 0.1, "Partial integration difference should be ~6.33, got {}", result);
    }
    
    #[test]
    fn test_higher_order_calculus() {
        // Second derivative: f''(x) = d/dx(d/dx(f(x)))
        let code = "
            let f = (λx.((* x) ((* x) x))) in  # f(x) = x³
            let f_prime = (λx.(diff f x)) in   # f'(x)
            (diff f_prime 2.0)                # f''(2)
        ";
        let result = eval_ok(code);
        // f(x) = x³, f'(x) = 3x², f''(x) = 6x, f''(2) = 12
        assert!((result - 12.0).abs() < 0.1, "Second derivative of x³ at x=2 should be ~12, got {}", result);
    }
    
    #[test]
    fn test_calculus_with_fuzzy_functions() {
        // Test differentiation/integration with fuzzy conditional functions
        let code = "
            let fuzzy_step = (λx.if (> x 0.0) then 1.0 else 0.0) in
            (diff fuzzy_step 0.0)
        ";
        let result = eval_ok(code);
        // At x=0, the fuzzy step function should have some finite derivative
        // (unlike a true step function which would be infinite)
        assert!(result.is_finite(), "Derivative of fuzzy step should be finite, got {}", result);
    }
    
    #[test]
    fn test_calculus_edge_cases() {
        // Test with functions that have zeros
        let code = "(integrate (λx.0.0) -5.0 5.0)";
        let result = eval_ok(code);
        assert!(result.abs() < 0.001, "Integral of zero function should be ~0, got {}", result);
        
        // Test with negative intervals
        let code = "(integrate (λx.x) 2.0 0.0)";
        let result = eval_ok(code);
        let expected = -2.0; // Should be negative of forward integral
        assert!((result - expected).abs() < 0.01, "Backwards integral should be ~{}, got {}", expected, result);
        
        // Test derivative at a point where function is not smooth
        let code = "(diff (abs) 0.0)";
        let result = eval_ok(code);
        // abs(x) is not differentiable at 0, but numerical diff should give something reasonable
        assert!(result.is_finite(), "Derivative of |x| at 0 should be finite (numerical), got {}", result);
    }
    
    #[test]
    fn test_calculus_composition() {
        // Chain rule: d/dx(f(g(x))) should work through numerical differentiation
        let code = "
            let f = (λx.((* x) x)) in      # f(x) = x²
            let g = (λx.((+ x) 1.0)) in    # g(x) = x + 1
            let composite = (λx.(f (g x))) in  # f(g(x)) = (x+1)²
            (diff composite 2.0)
        ";
        let result = eval_ok(code);
        // f(g(x)) = (x+1)², f'(g(x)) = 2(x+1), at x=2: f'(g(2)) = 2(3) = 6
        assert!((result - 6.0).abs() < 0.1, "Chain rule result should be ~6, got {}", result);
    }    
}