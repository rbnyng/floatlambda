// tests/tco_tests.rs

use float_lambda::{parse, Heap};
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

fn eval_ok(input: &str) -> f64 {
    let term = parse(input).unwrap();
    let mut heap = Heap::new();
    term.eval(&Rc::new(HashMap::new()), &mut heap).unwrap()
}

fn eval_with_timeout(input: &str, timeout_ms: u64) -> Result<f64, String> {
    let start = Instant::now();
    let result = eval_ok(input);
    let duration = start.elapsed();
    
    if duration.as_millis() > timeout_ms as u128 {
        Err(format!("Evaluation took too long: {}ms", duration.as_millis()))
    } else {
        Ok(result)
    }
}

#[cfg(test)]
mod tco_tests {
    use super::*;

    #[test]
    fn test_simple_tail_recursive_factorial() {
        let factorial_tail = "
            let rec fact_tail = (λn. λacc.
                if (< n 2) then acc
                else (fact_tail (- n 1) (* n acc))
            ) in
            (fact_tail 10 1)
        ";
        assert_eq!(eval_ok(factorial_tail), 3628800.0);
    }

    #[test]
    fn test_tail_recursive_countdown() {
        // This should complete quickly with TCO, slowly/crash without it
        let countdown = "
            let rec countdown = (λn.
                if (< n 1) then 0
                else (countdown (- n 1))
            ) in
            (countdown 5000)
        ";
        
        // Should complete in reasonable time (< 500ms) with TCO
        match eval_with_timeout(countdown, 500) {
            Ok(result) => assert_eq!(result, 0.0),
            Err(msg) => panic!("TCO failed: {}", msg),
        }
    }

    #[test]
    fn test_tail_recursive_fibonacci() {
        let fib_tail = "
            let rec fib_helper = (λn. λa. λb.
                if (< n 1) then a
                else (fib_helper (- n 1) b (+ a b))
            ) in
            let fib = (λn. (fib_helper n 0 1)) in
            (fib 25)
        ";
        assert_eq!(eval_ok(fib_tail), 75025.0);
    }

    #[test] 
    fn test_very_deep_tail_recursion() {
        // This will definitely stack overflow without proper TCO
        let deep_recursion = "
            let rec sum_to_n = (λn. λacc.
                if (< n 1) then acc
                else (sum_to_n (- n 1) (+ acc n))
            ) in
            (sum_to_n 10000 0)
        ";
        
        let result = eval_with_timeout(deep_recursion, 1000);
        match result {
            Ok(val) => {
                // Sum of 1 to 10000 = 10000 * 10001 / 2 = 50005000
                assert_eq!(val, 50005000.0);
            }
            Err(msg) => panic!("Deep tail recursion failed: {}", msg),
        }
    }

    #[test]
    fn test_tail_recursive_list_sum() {
        let list_sum = "
            let rec make_list = (λn.
                if (< n 1) then nil
                else (cons n (make_list (- n 1)))
            ) in
            let rec sum_list = (λlst. λacc.
                if (eq? lst nil) then acc
                else (sum_list (cdr lst) (+ acc (car lst)))
            ) in
            let mylist = (make_list 1000) in
            (sum_list mylist 0)
        ";
        
        let result = eval_with_timeout(list_sum, 2000);
        match result {
            Ok(val) => {
                // Sum of 1 to 1000 = 1000 * 1001 / 2 = 500500
                assert_eq!(val, 500500.0);
            }
            Err(msg) => panic!("Tail recursive list processing failed: {}", msg),
        }
    }

    #[test]
    fn test_mutually_recursive_even_odd() {
        let mutual_recursion = "
            let rec funcs = 
                (cons (λn. if (eq? n 0) then 1 else ((car (cdr funcs)) (- n 1)))  # is_even
                (cons (λn. if (eq? n 0) then 0 else ((car funcs) (- n 1)))      # is_odd
                      nil))
            in
            let is_even = (car funcs) in
            (is_even 10000)
        ";
        
        let result = eval_with_timeout(mutual_recursion, 1000);
        match result {
            Ok(val) => assert_eq!(val, 1.0), // 10000 is even
            Err(msg) => panic!("Mutually recursive functions failed: {}", msg),
        }
    }
    
    #[test]
    fn test_tail_call_through_higher_order_function() {
        let higher_order_tail = "
            let rec apply_n_times = (λf. λn. λx.
                if (< n 1) then x
                else (apply_n_times f (- n 1) (f x))
            ) in
            let inc = (+ 1) in
            (apply_n_times inc 5000 0)
        ";
        
        let result = eval_with_timeout(higher_order_tail, 1000);
        match result {
            Ok(val) => assert_eq!(val, 5000.0),
            Err(msg) => panic!("Higher order tail calls failed: {}", msg),
        }
    }

    #[test]
    fn test_non_tail_recursion_still_works() {
        // This should work but be slower (and limited in depth)
        let non_tail_factorial = "
            let rec factorial = (λn.
                if (< n 2) then 1
                else (* n (factorial (- n 1)))
            ) in
            (factorial 10)
        ";
        assert_eq!(eval_ok(non_tail_factorial), 3628800.0);
    }

    #[test]
    fn test_mixed_tail_and_non_tail() {
        // Test that mixing tail and non-tail recursion works correctly
        let mixed = "
            let rec tail_sum = (λn. λacc.
                if (< n 1) then acc
                else (tail_sum (- n 1) (+ acc n))
            ) in
            let rec non_tail_factorial = (λn.
                if (< n 2) then 1
                else (* n (non_tail_factorial (- n 1)))
            ) in
            let sum_result = (tail_sum 100 0) in
            let fact_result = (non_tail_factorial 5) in
            (+ sum_result fact_result)
        ";
        
        // Sum of 1 to 100 = 5050, 5! = 120, total = 5170
        assert_eq!(eval_ok(mixed), 5170.0);
    }

    #[test]
    fn test_tail_calls_in_let_expressions() {
        // Test that tail calls work when the recursive call is in a let body
        let let_tail = "
            let rec countdown_let = (λn.
                let next_n = (- n 1) in
                if (< next_n 0) then n
                else (countdown_let next_n)
            ) in
            (countdown_let 1000)
        ";
        
        let result = eval_with_timeout(let_tail, 500);
        match result {
            Ok(val) => assert_eq!(val, 0.0),
            Err(msg) => panic!("Tail calls in let expressions failed: {}", msg),
        }
    }

    #[test]
    fn test_performance_comparison() {
        // Compare performance of tail vs non-tail recursion
        let tail_version = "
            let rec sum_tail = (λn. λacc.
                if (< n 1) then acc
                else (sum_tail (- n 1) (+ acc n))
            ) in
            (sum_tail 1000 0)
        ";
        
        let start = Instant::now();
        let result = eval_ok(tail_version);
        let tail_duration = start.elapsed();
        
        assert_eq!(result, 500500.0);
        
        // Tail version should complete quickly
        assert!(tail_duration.as_millis() < 100, 
                "Tail recursion too slow: {}ms", tail_duration.as_millis());
    }

    #[test] 
    fn test_stack_safety() {
        // This test specifically checks that we don't get stack overflow
        // We'll use a very deep recursion that would definitely overflow without TCO
        let stack_safety_test = "
            let rec deep_identity = (λn. λx.
                if (< n 1) then x
                else (deep_identity (- n 1) x)
            ) in
            (deep_identity 50000 42)
        ";
        
        let result = eval_with_timeout(stack_safety_test, 2000);
        match result {
            Ok(val) => assert_eq!(val, 42.0),
            Err(msg) => panic!("Stack safety test failed: {}", msg),
        }
    }

    #[test]
    fn test_tail_recursion_with_complex_expressions() {
        // Test TCO with more complex expressions in tail position
        let complex_tail = "
            let rec complex_sum = (λlst. λmultiplier. λacc.
                if (eq? lst nil) then acc
                else 
                    let head = (car lst) in
                    let tail = (cdr lst) in
                    let new_acc = (+ acc (* head multiplier)) in
                    (complex_sum tail multiplier new_acc)
            ) in
            let test_list = (cons 1 (cons 2 (cons 3 (cons 4 (cons 5 nil))))) in
            (complex_sum test_list 10 0)
        ";
        
        // (1 + 2 + 3 + 4 + 5) * 10 = 150
        assert_eq!(eval_ok(complex_tail), 150.0);
    }

    #[test]
    fn test_nested_tail_calls() {
        // Use the recursive list pattern for mutual recursion.
        let nested_tail = "
            let rec helpers =
                (cons (λn. if (< n 1) then 0 else ((car (cdr helpers)) (- n 1)))  # helper1
                (cons (λn. if (< n 1) then 0 else ((car helpers) (- n 1)))      # helper2
                      nil))
            in
            let helper1 = (car helpers) in
            (helper1 10000)
        ";
        
        let result = eval_with_timeout(nested_tail, 1000);
        match result {
            Ok(val) => assert_eq!(val, 0.0),
            Err(msg) => panic!("Nested tail calls failed: {}", msg),
        }
    }
}

// Benchmark tests (run with --ignored flag)
#[cfg(test)]
mod tco_benchmarks {
    use super::*;

    #[test]
    #[ignore]
    fn benchmark_tail_recursion_vs_iteration() {
        println!("\n=== TCO Performance Benchmarks ===");
        
        let test_cases = vec![
            ("Small (n=100)", 100),
            ("Medium (n=1000)", 1000), 
            ("Large (n=10000)", 10000),
            ("Very Large (n=100000)", 100000),
        ];
        
        for (name, n) in test_cases {
            let tail_recursive = format!("
                let rec sum_tail = (λn. λacc.
                    if (< n 1) then acc
                    else (sum_tail (- n 1) (+ acc n))
                ) in
                (sum_tail {} 0)
            ", n);
            
            let start = Instant::now();
            let result = eval_ok(&tail_recursive);
            let duration = start.elapsed();
            
            let expected = (n * (n + 1)) / 2;
            assert_eq!(result, expected as f64);
            
            println!("{}: {}ms (result: {})", name, duration.as_millis(), result);
        }
    }

    #[test]
    #[ignore] 
    fn stress_test_extreme_depth() {
        println!("\n=== Extreme Depth Stress Test ===");
        
        let extreme_depth = "
            let rec extreme_countdown = (λn.
                if (< n 1) then 0
                else (extreme_countdown (- n 1))
            ) in
            (extreme_countdown 1000000)
        ";
        
        let start = Instant::now();
        let result = eval_ok(extreme_depth);
        let duration = start.elapsed();
        
        assert_eq!(result, 0.0);
        println!("1 million tail calls: {}ms", duration.as_millis());
        
        // Should complete in reasonable time even with 1M calls
        assert!(duration.as_secs() < 10, "Too slow for 1M tail calls: {}s", duration.as_secs());
    }
}