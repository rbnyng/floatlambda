# FloatLambda

FloatLambda is a functional programming language built on a simple constraint: **every value is an IEEE 754 double-precision floating-point number (f64)**.

## Core Philosophy

Most programming languages have complex type systems to prevent errors. FloatLambda takes the opposite approach: achieve type safety by having only one type. There are no type errors because there's only one type to work with.

## Basic Values

Everything in FloatLambda is represented as an f64:

```
42.0        # Numbers are straightforward
1.0         # True (any non-zero value)
0.0         # False
nil         # Represented as negative infinity
```

Functions and data structures are stored on a heap and referenced through special NaN-boxed f64 values but they're still just f64s that happen to be callable or indexable.

## Syntax Basics

FloatLambda uses lambda calculus syntax with some modern conveniences:

### Lambda Functions
```
λx.x                   # Identity function
λx.(+ x 1)             # Add one to x
λf.λx.(f x)            # Function application
```

### Function Application
```
(λx.x 42)              # Apply identity to 42 → 42.0
((+ 1) 2)              # Curried addition → 3.0
(+ 1 2)                # Multiple arguments → 3.0 (parsed as ((+ 1) 2))
```

### Let Bindings
```
let x = 10 in (+ x 5)                   # Simple binding → 15.0
let f = λx.(* x 2) in (f 21)            # Function binding → 42.0
let rec fac = λn.if (< n 2) then 1 
                 else (* n (fac (- n 1))) 
    in (fac 5)                           # Recursion → 120.0
```

## Fuzzy Logic

The unique feature of FloatLambda is its **continuous truth values**. Instead of binary true/false, conditions can be anywhere between 0.0 and 1.0, leading to blended execution:

### Fuzzy Conditionals
```
if 0.7 then 100 else 200                # → 130.0
# Calculation: 0.7 * 100 + 0.3 * 200 = 70 + 60 = 130
```

### Fuzzy Logic Operators
```
(fuzzy_and 0.8 0.6)                     # → 0.48 (multiplication)
(fuzzy_or 0.8 0.6)                      # → 0.92 (probabilistic OR)
(fuzzy_not 0.3)                         # → 0.7 (1 - x)
```

### Fuzzy Equality
```
(== 1.0 1.1)                           # → ~0.905 (similarity-based)
(== 1.0 1.0)                           # → 1.0 (exact match)
```

## Data Structures: Lists

Lists are built using cons cells, like in Lisp:

```
nil                                     # Empty list
(cons 1 nil)                           # List with one element: [1]
(cons 1 (cons 2 (cons 3 nil)))        # List: [1, 2, 3]

# List operations
(car (cons 1 2))                       # → 1.0 (first element)
(cdr (cons 1 2))                       # → 2.0 (rest)
(length (cons 1 (cons 2 nil)))         # → 2.0
```

## Higher-Order Functions

FloatLambda supports functional programming patterns:

```
# Map function
let add10 = (+ 10) in
let mylist = (cons 1 (cons 2 (cons 3 nil))) in
(map add10 mylist)                      # → [11, 12, 13]

# Filter function  
let is_positive = (> 0) in
let mylist = (cons -1 (cons 2 (cons -3 (cons 4 nil)))) in
(filter is_positive mylist)            # → [2, 4]

# Fold (reduce)
let mylist = (cons 1 (cons 2 (cons 3 nil))) in
(foldl + 0 mylist)                     # → 6.0 (sum of list)
```

## Text Processing

Strings are represented as lists of character codes:

```
# "Hi" is represented as:
(cons 72.0 (cons 105.0 nil))          # [72, 105] → "Hi"

# The print function outputs characters:
(print (cons 72.0 (cons 105.0 nil)))  # Prints: Hi
```

## Probabilistic Features

FloatLambda includes some weird probabilistic behaviors:

### Probabilistic Character Output
The `print` function uses fractional parts as rendering probabilities:
```
(print (cons 65.7 nil))               # 70% chance of 'B', 30% chance of 'A'
```

## Example Programs

### Factorial
```
let rec factorial = λn.
    if (< n 2) then 1 
    else (* n (factorial (- n 1)))
in (factorial 5)                       # → 120.0
```

### List Processing
```
let rec sum_list = λl.
    if (eq? l nil) then 0
    else (+ (car l) (sum_list (cdr l)))
in 
let mylist = (cons 10 (cons 20 (cons 30 nil))) in
(sum_list mylist)                      # → 60.0
```

### Higher-Order Function Example
```
let compose = λf.λg.λx.(f (g x)) in
let add1 = (+ 1) in
let mul2 = (* 2) in
let add1_then_mul2 = (compose mul2 add1) in
(add1_then_mul2 5)                     # → 12.0 (5+1)*2
```

### Fuzzy Decision Making
```
let confidence = 0.8 in
let safe_action = 10 in
let risky_action = 100 in
if confidence then safe_action else risky_action    # → 28.0
# Blends: 0.8 * 10 + 0.2 * 100 = 8 + 20 = 28
```

## Built-in Functions

### Arithmetic
- `+`, `-`, `*`, `/` - Basic arithmetic
- `neg`, `abs`, `sqrt` - Unary math functions
- `min`, `max` - Comparison functions
- `div`, `rem` - Integer division and remainder (this is like % and // from Python)

### Comparison  
- `<`, `>`, `<=`, `>=` - Numeric comparisons (return 1.0 or 0.0)
- `==` - Fuzzy equality (returns similarity score)
- `eq?` - Strict equality (exact bit-wise comparison)

### Logic
- `fuzzy_and`, `fuzzy_or`, `fuzzy_not` - Continuous logic operations

### Lists
- `cons`, `car`, `cdr` - List construction and access
- `length` - List length
- `map`, `filter`, `foldl` - Higher-order list operations

### I/O
- `print` - Output a list of character codes
- `read-char` - Read a single character (returns character code)
- `read-line` - Read a line of text (returns list of character codes)

## What the Fuck?

Everything is f64, trust.

## Getting Started

FloatLambda includes a REPL:

```bash
$ float_lambda
FloatLambda v3 REPL
Enter expressions, 'quit', or ':examples'
> ((+ 1) 2)
Parsed: (+ 1 2)
Result: 3.0 (1 objects alive)
> let x = 42 in (+ x 8)
Parsed: let x = 42 in (+ x 8)  
Result: 50.0 (1 objects alive)
```
