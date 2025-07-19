# FloatLambda

FloatLambda is a functional programming language built on a simple constraint: **every value is an IEEE 754 double-precision floating-point number (f64)**.

## Core Philosophy

Most programming languages have complex type systems to prevent errors. FloatLambda takes the opposite approach: achieve type safety by having only one type. Traditional type errors don't exist; instead, they transmute into weird, wonderful, and whimsical runtime behavior. Take that, OCaml bros.

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

FloatLambda uses lambda calculus syntax (λ or \\) with some modern conveniences:

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

# Any binary choice can be made continuous
let continuous_choice = (λcondition. λoption_a. λoption_b.
  if condition then option_a else option_b
)
```

Works for:
 - Values: (continuous_choice 0.7 100 200)
 - Functions: (continuous_choice 0.7 (λx.x) (λx.(* x 2)))
 - Evaluations: (continuous_choice 0.7 (eval expr_a) (eval expr_b))
 - Transformations: (continuous_choice 0.7 (transform expr) expr)

### Fuzzy Logic Operators
```
(fuzzy_and 0.8 0.6)                     # → 0.48 (multiplication)
(fuzzy_or 0.8 0.6)                      # → 0.92 (probabilistic OR)
(fuzzy_not 0.3)                         # → 0.7 (1 - x)
```

### Fuzzy Equality

The `==` operator is scale-invariant, meaning it compares numbers based on their relative difference.

```
(== 1.0 1.1)                             # → ~0.913 (9% difference)
(== 1000.0 1001.0)                       # → ~0.999 (0.1% difference)
(== 1.0 1.0)                             # → 1.0
```

It uses the formula $e^{- \\frac{|x-y|}{\\max(|x|, |y|, 1.0)}}$. For strict, bit-wise equality, use `eq?`.

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

## Calculus

Since everything is an f64, every operation is differentiable. FloatLambda provides built-in calculus operations:

### Derivatives
```
# Derivative of x² at x=5 (should be 10)
let f = λx.(* x x) in
(diff f 5.0)                           # → 10.0

# Derivative of sin approximation
let sin_approx = λx.(- x (/ (* x (* x x)) 6)) in
(diff sin_approx 0.0)                 # → ~1.0 (cos(0))
```

### Integration
```
# Definite integral of x² from 0 to 3 (should be 9)
let f = λx.(* x x) in
(integrate f 0.0 3.0)                 # → 9.0

# Curried integration for reuse
let integrate_from_zero = (integrate f 0.0) in
let area1 = (integrate_from_zero 2.0) in
let area2 = (integrate_from_zero 4.0) in
(- area2 area1)                       # Area between x=2 and x=4
```

### Fundamental Theorem of Calculus
```
# Verify that differentiation and integration are inverses
let f = λx.(* x x) in
let F = λx.(integrate f 0.0 x) in     # Antiderivative
(diff F 5.0)                          # Should equal f(5) = 25.0
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
- `+`, `-`, `*`, `/` - Basic arithmetic (curried: `(+ 1 2)` or `((+ 1) 2)`)
- `neg`, `abs` - Unary operations: negate, absolute value
- `sqrt` - Square root (returns NaN for negative inputs)
- `min`, `max` - Minimum and maximum of two values
- `div`, `rem` - Integer division and remainder (this is like % and // from Python)
- `exp` - Exponential function (e^x)

### Mathematics
- `sin`, `cos`, `tan` - Trigonometric functions
- `asin`, `acos`, `atan`, `atan2` - Inverse trigonometric functions
- `sinh`, `cosh`, `tanh` - Hyperbolic functions
- `asinh`, `acosh`, `atanh` - Inverse hyperbolic functions
- `log`, `log2`, `log10` - Natural, base-2, and base-10 logarithms
- `exp2` - Powers of 2 (2^x)
- `pow` - General exponentiation: `(pow base exponent)`
- `cbrt` - Cube root
- `floor`, `ceil`, `round`, `trunc` - Rounding functions
- `fract` - Fractional part
- `signum` - Sign function (-1, 0, or 1)

### Special Functions
- `gamma` - Gamma function (generalized factorial)
- `lgamma` - Natural logarithm of gamma function
- `erf`, `erfc` - Error function and complementary error function
- `hypot` - Euclidean distance: `sqrt(x² + y²)`
- `copysign` - Copy sign from one number to another
- `degrees`, `radians` - Convert between radians and degrees

### Mathematical Constants
- `pi` - π (3.14159...)
- `e` - Euler's number (2.71828...)
- `tau` - τ = 2π (6.28318...)
- `sqrt2` - √2 (1.41421...)
- `ln2` - ln(2) (0.69314...)
- `ln10` - ln(10) (2.30258...)

### Floating Point Utilities
- `is_nan` - Check if value is NaN (returns 1.0 or 0.0)
- `is_infinite` - Check if value is infinite
- `is_finite` - Check if value is finite
- `is_normal` - Check if value is a normal floating point number

### Randomness
- `random` - Random float in [0, 1)
- `random_range` - Random float in specified range: `(random_range min max)`
- `random_normal` - Normal distribution: `(random_normal mean std_dev)`

### Comparison  
- `<`, `>`, `<=`, `>=` - Numeric comparisons (return 1.0 or 0.0)
- `==` - Fuzzy equality (returns similarity score: `e^(-|x-y|)`)
- `eq?` - Strict equality (exact bit-wise comparison, handles NaN correctly)

### Calculus
- `diff` - Numerical differentiation: `(diff f x)` computes f'(x)
- `integrate` - Numerical integration: `(integrate f a b)` computes ∫ₐᵇ f(x)dx
  - Curried form: `((integrate f a) b)` allows partial application

### Fuzzy Logic
- `fuzzy_and` - Continuous AND: `(fuzzy_and 0.8 0.6)` → 0.48
- `fuzzy_or` - Continuous OR: `(fuzzy_or 0.8 0.6)` → 0.92  
- `fuzzy_not` - Continuous NOT: `(fuzzy_not 0.3)` → 0.7

### Lists
- `cons`, `car`, `cdr` - List construction and access
- `length` - List length
- `map` - Apply function to each element: `(map f list)`
- `filter` - Keep elements matching predicate: `(filter predicate list)`
- `foldl` - Left fold/reduce: `(foldl function initial list)`

### I/O
- `print` - Output a list of character codes (with probabilistic rendering)
- `read-char` - Read a single character (returns character code)
- `read-line` - Read a line of text (returns list of character codes)

### Machine Learning (Tensor Operations)
- `tensor` - Create tensor: `(tensor shape_list data_list)`
- `add_t` - Element-wise tensor addition
- `matmul` - Matrix multiplication  
- `sigmoid_t` - Sigmoid activation function
- `reshape` - Reshape tensor: `(reshape tensor new_shape)`
- `transpose` - Transpose 2D tensor
- `sum_t`, `mean_t` - Reduce tensor to scalar
- `get_data`, `get_shape`, `get_grad` - Extract tensor information
- `grad` - Automatic differentiation: `(grad function input_tensor)`

### Examples

```
# Mathematical computation
(sin (/ pi 2))                        # → 1.0
(pow 2 10)                            # → 1024.0
(gamma 5)                             # → 24.0 (4!)

# Fuzzy logic
(fuzzy_and 0.8 0.6)                   # → 0.48
(== 1.0 1.1)                          # → ~0.905

# Calculus
let f = λx.(* x x) in
(diff f 5.0)                          # → 10.0 (derivative of x²)
(integrate f 0.0 3.0)                 # → 9.0 (integral of x²)

# Machine learning
let weights = (tensor (cons 2 (cons 1 nil)) (cons 0.5 (cons -0.3 nil))) in
let gradients = (grad (λw. (sum_t w)) weights) in
(get_data gradients)                  # → [1.0, 1.0]

# Randomness
(random_normal 0 1)                   # → ~0.123 (different each time)
(random_range 10 20)                  # → ~15.67 (between 10 and 20)
```

## What the Fuck?

Everything is f64, trust.

## Getting Started

FloatLambda includes a REPL for interactive exploration:

```bash
$ float_lambda
FloatLambda v3 REPL
Enter expressions, 'quit', ':examples', or ':inspect <id>'
> ((+ 1) 2)
Parsed: (+ 1 2)
Result: 3.0 (1 objects alive)
> let x = 42 in (+ x 8)
Parsed: let x = 42 in (+ x 8)  
Result: 50.0 (1 objects alive)
```

Or run scripts directly:

```bash
$ float_lambda factorial.fl
```

Example script (`factorial.fl`):
```
# Calculate factorial of 10
let rec factorial = λn.
    if (< n 2) then 1 
    else (* n (factorial (- n 1)))
in 
let result = (factorial 10) in
(print "Result: ")
(print result)
```

The `:examples` command in the REPL shows more usage patterns.
