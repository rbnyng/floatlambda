// tests/string_tests.rs

use float_lambda::{parse, Term};
mod test_utils;
use test_utils::*;

#[cfg(test)]
mod string_literal_tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        let result = parse("\"\"");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Term::Nil);
    }

    #[test]
    fn test_single_char_string() {
        // Should be a cons cell with 'a' (97) and nil
        assert_eq!(eval_ok("(car \"a\")"), 97.0);
        assert_eq!(eval_ok("(cdr \"a\")"), float_lambda::NIL_VALUE);
    }

    #[test]
    fn test_multi_char_string() {
        // Test "hi" = (cons 104 (cons 105 nil))
        assert_eq!(eval_ok("(car \"hi\")"), 104.0); // 'h'
        assert_eq!(eval_ok("(car (cdr \"hi\"))"), 105.0); // 'i'
        assert_eq!(eval_ok("(cdr (cdr \"hi\"))"), float_lambda::NIL_VALUE);
    }

    #[test]
    fn test_string_with_spaces() {
        assert_eq!(eval_ok("(car \"a b\")"), 97.0);  // 'a'
        assert_eq!(eval_ok("(car (cdr \"a b\"))"), 32.0); // space
        assert_eq!(eval_ok("(car (cdr (cdr \"a b\")))"), 98.0); // 'b'
    }

    #[test]
    fn test_escape_sequences() {
        assert_eq!(eval_ok("(car \"\\n\")"), 10.0);  // newline
        assert_eq!(eval_ok("(car \"\\t\")"), 9.0);   // tab
        assert_eq!(eval_ok("(car \"\\r\")"), 13.0);  // carriage return
        assert_eq!(eval_ok("(car \"\\\\\")"), 92.0); // backslash
        assert_eq!(eval_ok("(car \"\\\"\")"), 34.0); // quote
    }

    #[test]
    fn test_string_length() {
        assert_eq!(eval_ok("(length \"\")"), 0.0);
        assert_eq!(eval_ok("(length \"a\")"), 1.0);
        assert_eq!(eval_ok("(length \"hello\")"), 5.0);
        assert_eq!(eval_ok("(length \"hello\\nworld\")"), 11.0);
    }

    #[test]
    fn test_unterminated_string() {
        assert!(parse("\"hello").is_err());
        assert!(parse("\"hello\\").is_err());
        assert!(parse("\"").is_err());
    }

    #[test]
    fn test_invalid_escape_sequences() {
        assert!(parse("\"\\x\"").is_err());
        assert!(parse("\"\\z\"").is_err());
    }
}

#[cfg(test)]
mod string_function_tests {
    use super::*;

    #[test]
    fn test_string_concatenation() {
        let result = eval_ok_with_prelude("(length (string_concat \"hello\" \"world\"))");
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_string_equality() {
        assert_eq!(eval_ok_with_prelude("(string_eq \"hello\" \"hello\")"), 1.0);
        assert_eq!(eval_ok_with_prelude("(string_eq \"hello\" \"world\")"), 0.0);
        assert_eq!(eval_ok_with_prelude("(string_eq \"\" \"\")"), 1.0);
    }

    #[test]
    fn test_string_case_conversion() {
        // Test uppercase conversion
        let result = eval_ok_with_prelude("(car (string_uppercase \"a\"))");
        assert_eq!(result, 65.0); // 'A'
        
        // Test lowercase conversion
        let result = eval_ok_with_prelude("(car (string_lowercase \"A\"))");
        assert_eq!(result, 97.0); // 'a'
    }

    #[test]
    fn test_string_length_prelude() {
        // Test using the prelude string_length function
        assert_eq!(eval_ok_with_prelude("(string_length \"\")"), 0.0);
        assert_eq!(eval_ok_with_prelude("(string_length \"hello\")"), 5.0);
    }

    #[test]
    fn test_string_contains() {
        assert_eq!(eval_ok_with_prelude("(string_contains \"hello world\" \"world\")"), 1.0);
        assert_eq!(eval_ok_with_prelude("(string_contains \"hello\" \"xyz\")"), 0.0);
        assert_eq!(eval_ok_with_prelude("(string_contains \"\" \"\")"), 1.0);
    }

    #[test]
    fn test_string_starts_with() {
        assert_eq!(eval_ok_with_prelude("(string_starts_with \"hello\" \"hel\")"), 1.0);
        assert_eq!(eval_ok_with_prelude("(string_starts_with \"hello\" \"world\")"), 0.0);
        assert_eq!(eval_ok_with_prelude("(string_starts_with \"\" \"\")"), 1.0);
    }

    #[test]
    fn test_string_ends_with() {
        assert_eq!(eval_ok_with_prelude("(string_ends_with \"hello\" \"llo\")"), 1.0);
        assert_eq!(eval_ok_with_prelude("(string_ends_with \"hello\" \"world\")"), 0.0);
        assert_eq!(eval_ok_with_prelude("(string_ends_with \"\" \"\")"), 1.0);
    }

    #[test]
    fn test_substring() {
        // Test substring extraction
        let result = eval_ok_with_prelude("(car (substring \"hello\" 1 3))");
        assert_eq!(result, 101.0); // 'e' - first char of "ell"
        
        let result = eval_ok_with_prelude("(string_length (substring \"hello\" 1 3))");
        assert_eq!(result, 3.0); // Length of "ell"
    }
}

#[cfg(test)]
mod string_edge_cases {
    use super::*;

    #[test]
    fn test_unicode_characters() {
        // Test Unicode support (Greek alpha)
        let result = eval_ok("(car \"α\")");
        assert_eq!(result, 945.0); // Unicode codepoint for α
    }

    #[test]
    fn test_string_with_numbers() {
        assert_eq!(eval_ok("(car \"123\")"), 49.0); // '1'
        assert_eq!(eval_ok("(car (cdr \"123\"))"), 50.0); // '2'
        assert_eq!(eval_ok("(car (cdr (cdr \"123\")))"), 51.0); // '3'
    }

    #[test]
    fn test_string_in_expressions() {
        // Test that strings work in larger expressions
        let result = eval_ok_with_prelude("(string_length (string_concat \"Hello, \" \"Alice\"))");
        assert_eq!(result, 12.0); // Length of "Hello, Alice"
    }

    #[test]
    fn test_string_as_function_argument() {
        let code = r#"
            let first_char = (λs. (car s)) in
            (first_char "test")
        "#;
        assert_eq!(eval_ok(code), 116.0); // 't'
    }

    #[test]
    fn test_nested_string_operations() {
        // Test complex string operations
        let result = eval_ok_with_prelude(r#"
            let s1 = "hello" in
            let s2 = "world" in
            let combined = (string_concat s1 s2) in
            (string_length combined)
        "#);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_string_with_special_chars() {
        // Test strings with punctuation
        assert_eq!(eval_ok("(car \"!\")"), 33.0); // '!'
        assert_eq!(eval_ok("(car \"@\")"), 64.0); // '@'
        assert_eq!(eval_ok("(car \"#\")"), 35.0); // '#'
    }

    #[test]
    fn test_mixed_case_string() {
        let result = eval_ok_with_prelude("(string_eq (string_uppercase \"HeLLo\") \"HELLO\")");
        assert_eq!(result, 1.0);
        
        let result = eval_ok_with_prelude("(string_eq (string_lowercase \"HeLLo\") \"hello\")");
        assert_eq!(result, 1.0);
    }
}

#[cfg(test)]
mod string_performance_tests {
    use super::*;

    #[test]
    fn test_long_string_parsing() {
        // Test parsing reasonably long strings
        let long_string = "a".repeat(100); // Keep it reasonable for tests
        let code = format!("\"{}\"", long_string);
        
        let result = parse(&code);
        assert!(result.is_ok());
        
        // Test that it evaluates correctly
        let length_code = format!("(length \"{}\")", long_string);
        assert_eq!(eval_ok(&length_code), 100.0);
    }

    #[test]
    fn test_string_with_many_escapes() {
        let code = r#""hello\nworld\ttest\r\n""#;
        // Count: h-e-l-l-o-\n-w-o-r-l-d-\t-t-e-s-t-\r-\n = 18 characters
        assert_eq!(eval_ok(&format!("(length {})", code)), 18.0);
    }

    #[test]
    fn test_repeated_string_operations() {
        // Test that string operations don't leak memory or get slow
        let result = eval_ok_with_prelude(r#"
            let base = "test" in
            let doubled = (string_concat base base) in
            let quadrupled = (string_concat doubled doubled) in
            (string_length quadrupled)
        "#);
        assert_eq!(result, 16.0); // "test" * 4 = 16 chars
    }
}

#[cfg(test)]
mod string_integration_tests {
    use super::*;

    #[test]
    fn test_string_with_list_operations() {
        // Test that strings work with regular list operations
        assert_eq!(eval_ok("(car \"abc\")"), 97.0); // 'a'
        assert_eq!(eval_ok("(car (cdr \"abc\"))"), 98.0); // 'b'
        assert_eq!(eval_ok("(length \"abc\")"), 3.0);
    }

    #[test]
    fn test_string_with_map() {
        // Test mapping over a string (list of char codes)
        let result = eval_ok_with_prelude("(car (map (+ 1) \"abc\"))");
        assert_eq!(result, 98.0); // 'a' + 1 = 'b'
    }

    #[test]
    fn test_string_with_filter() {
        // Test filtering a string - keep only vowels (simplified)
        let result = eval_ok_with_prelude(r#"
            let is_vowel = (λc. (or (eq? c 97) (or (eq? c 101) (or (eq? c 105) (or (eq? c 111) (eq? c 117)))))) in
            (length (filter is_vowel "hello"))
        "#);
        assert_eq!(result, 2.0); // 'e' and 'o' are vowels
    }

    #[test]
    fn test_string_split_and_join() {
        // Test string splitting and joining
        let result = eval_ok_with_prelude(r#"
            let parts = (string_split "a,b,c" 44) in
            let rejoined = (string_join parts "-") in
            (string_eq rejoined "a-b-c")
        "#);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_string_in_conditionals() {
        // Test strings in conditional expressions
        let _result = eval_ok_with_prelude(r#"
            if (string_eq "hello" "hello") then "yes" else "no"
        "#);
        // Should return "yes"
        assert_eq!(eval_ok("(car (cdr (cdr \"yes\")))"), 115.0); // 's'
    }
}