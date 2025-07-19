// src/math.rs

use crate::error::EvalError;

// This module houses stateless, pure math functions.

use rand::Rng;

// Mathematical constants
pub const PI: f64 = std::f64::consts::PI;
pub const E: f64 = std::f64::consts::E;
pub const TAU: f64 = std::f64::consts::TAU; // 2π
pub const SQRT_2: f64 = std::f64::consts::SQRT_2;
pub const LN_2: f64 = std::f64::consts::LN_2;
pub const LN_10: f64 = std::f64::consts::LN_10;

// --- Private Helpers ---

// Helper for gamma function (Lanczos approximation)
fn gamma_lanczos(z: f64) -> f64 {
    if z < 0.5 {
        // Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
        PI / (PI * z).sin() / gamma_lanczos(1.0 - z)
    } else {
        // Lanczos coefficients for g=7
        const G: f64 = 7.0;
        const COEFFS: [f64; 9] = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        
        let z = z - 1.0;
        let mut x = COEFFS[0];
        for i in 1..COEFFS.len() {
            x += COEFFS[i] / (z + i as f64);
        }
        
        let t = z + G + 0.5;
        (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
    }
}

// Error function approximation (Abramowitz and Stegun)
fn erf_approx(x: f64) -> f64 {
    if x == 0.0 { return 0.0; }
    
    let sign = if x > 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    
    // Constants for approximation
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;
    
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    
    sign * y
}

// --- Public, Pure Functions ---

// Nullary
pub fn fl_random() -> f64 { rand::thread_rng().gen::<f64>() }
pub fn fl_random_range(min: f64, max: f64) -> f64 { rand::thread_rng().gen_range(min..max) }
pub fn fl_random_normal(mean: f64, std_dev: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let u1: f64 = rng.gen();
    let u2: f64 = rng.gen();
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    mean + std_dev * z0
}

// Unary
pub fn fl_sin(x: f64) -> f64 { x.sin() }
pub fn fl_cos(x: f64) -> f64 { x.cos() }
pub fn fl_tan(x: f64) -> f64 { x.tan() }
pub fn fl_exp(x: f64) -> f64 { x.exp() }
pub fn fl_log(x: f64) -> f64 { if x <= 0.0 { f64::NAN } else { x.ln() } }
pub fn fl_asin(x: f64) -> f64 { if x < -1.0 || x > 1.0 { f64::NAN } else { x.asin() } }
pub fn fl_acos(x: f64) -> f64 { if x < -1.0 || x > 1.0 { f64::NAN } else { x.acos() } }
pub fn fl_atan(x: f64) -> f64 { x.atan() }
pub fn fl_sinh(x: f64) -> f64 { x.sinh() }
pub fn fl_cosh(x: f64) -> f64 { x.cosh() }
pub fn fl_tanh(x: f64) -> f64 { x.tanh() }
pub fn fl_asinh(x: f64) -> f64 { x.asinh() }
pub fn fl_acosh(x: f64) -> f64 { if x < 1.0 { f64::NAN } else { x.acosh() } }
pub fn fl_atanh(x: f64) -> f64 { if x <= -1.0 || x >= 1.0 { f64::NAN } else { x.atanh() } }
pub fn fl_sqrt(x: f64) -> f64 { if x < 0.0 { f64::NAN } else { x.sqrt() } }
pub fn fl_cbrt(x: f64) -> f64 { x.cbrt() }
pub fn fl_exp2(x: f64) -> f64 { x.exp2() }
pub fn fl_log2(x: f64) -> f64 { if x <= 0.0 { f64::NAN } else { x.log2() } }
pub fn fl_log10(x: f64) -> f64 { if x <= 0.0 { f64::NAN } else { x.log10() } }
pub fn fl_floor(x: f64) -> f64 { x.floor() }
pub fn fl_ceil(x: f64) -> f64 { x.ceil() }
pub fn fl_round(x: f64) -> f64 { x.round() }
pub fn fl_trunc(x: f64) -> f64 { x.trunc() }
pub fn fl_fract(x: f64) -> f64 { x.fract() }
pub fn fl_signum(x: f64) -> f64 { x.signum() }
pub fn fl_gamma(x: f64) -> f64 { if x <= 0.0 && x.fract() == 0.0 { f64::NAN } else { gamma_lanczos(x) } }
pub fn fl_lgamma(x: f64) -> f64 { if x <= 0.0 && x.fract() == 0.0 { f64::NAN } else { gamma_lanczos(x).ln() } }
pub fn fl_erf(x: f64) -> f64 { erf_approx(x) }
pub fn fl_erfc(x: f64) -> f64 { 1.0 - erf_approx(x) }
pub fn fl_degrees(x: f64) -> f64 { x * 180.0 / PI }
pub fn fl_radians(x: f64) -> f64 { x * PI / 180.0 }
pub fn fl_is_nan(x: f64) -> f64 { if x.is_nan() { 1.0 } else { 0.0 } }
pub fn fl_is_infinite(x: f64) -> f64 { if x.is_infinite() { 1.0 } else { 0.0 } }
pub fn fl_is_finite(x: f64) -> f64 { if x.is_finite() { 1.0 } else { 0.0 } }
pub fn fl_is_normal(x: f64) -> f64 { if x.is_normal() { 1.0 } else { 0.0 } }

// Binary
pub fn fl_pow(x: f64, y: f64) -> f64 { x.powf(y) }
pub fn fl_atan2(y: f64, x: f64) -> f64 { y.atan2(x) }
pub fn fl_hypot(x: f64, y: f64) -> f64 { x.hypot(y) }
pub fn fl_copysign(x: f64, y: f64) -> f64 { x.copysign(y) }


// --- Public API for the interpreter ---

pub fn execute_math_builtin(op: &str, args: &[f64]) -> Result<f64, EvalError> {
    match op {
        // Basic transcendental functions
        "sin" => Ok(fl_sin(args[0])),
        "cos" => Ok(fl_cos(args[0])),
        "tan" => Ok(fl_tan(args[0])),
        "exp" => Ok(fl_exp(args[0])),
        "log" => Ok(fl_log(args[0])),
        
        // Inverse trig functions
        "asin" => Ok(fl_asin(args[0])),
        "acos" => Ok(fl_acos(args[0])),
        "atan" => Ok(fl_atan(args[0])),
        "atan2" => Ok(fl_atan2(args[0], args[1])),
        
        // Hyperbolic functions
        "sinh" => Ok(fl_sinh(args[0])),
        "cosh" => Ok(fl_cosh(args[0])),
        "tanh" => Ok(fl_tanh(args[0])),
        "asinh" => Ok(fl_asinh(args[0])),
        "acosh" => Ok(fl_acosh(args[0])),
        "atanh" => Ok(fl_atanh(args[0])),
        
        // Power and root functions
        "pow" => Ok(fl_pow(args[0], args[1])),
        "sqrt" => Ok(fl_sqrt(args[0])),
        "cbrt" => Ok(fl_cbrt(args[0])),
        "exp2" => Ok(fl_exp2(args[0])),
        "log2" => Ok(fl_log2(args[0])),
        "log10" => Ok(fl_log10(args[0])),
        
        // Rounding and utility functions
        "floor" => Ok(fl_floor(args[0])),
        "ceil" => Ok(fl_ceil(args[0])),
        "round" => Ok(fl_round(args[0])),
        "trunc" => Ok(fl_trunc(args[0])),
        "fract" => Ok(fl_fract(args[0])),
        "signum" => Ok(fl_signum(args[0])),
        
        // Special functions
        "gamma" => Ok(fl_gamma(args[0])),
        "lgamma" => Ok(fl_lgamma(args[0])),
        "erf" => Ok(fl_erf(args[0])),
        "erfc" => Ok(fl_erfc(args[0])),
        
        // Random number generation
        "random" => Ok(fl_random()),
        "random_range" => Ok(fl_random_range(args[0], args[1])),
        "random_normal" => Ok(fl_random_normal(args[0], args[1])),
        
        // Mathematical constants
        "pi" => Ok(PI),
        "e" => Ok(E),
        "tau" => Ok(TAU),
        "sqrt2" => Ok(SQRT_2),
        "ln2" => Ok(LN_2),
        "ln10" => Ok(LN_10),
        
        // Additional utility functions
        "degrees" => Ok(fl_degrees(args[0])),
        "radians" => Ok(fl_radians(args[0])),
        "hypot" => Ok(fl_hypot(args[0], args[1])),
        "copysign" => Ok(fl_copysign(args[0], args[1])),
        
        // Floating point utilities
        "is_nan" => Ok(fl_is_nan(args[0])),
        "is_infinite" => Ok(fl_is_infinite(args[0])),
        "is_finite" => Ok(fl_is_finite(args[0])),
        "is_normal" => Ok(fl_is_normal(args[0])),
        
        _ => Err(EvalError::TypeError(format!("Unknown math builtin: {}", op))),
    }
}

pub fn get_math_builtin_arity(op: &str) -> Option<usize> {
    match op {
        // Nullary (constants)
        "pi" | "e" | "tau" | "sqrt2" | "ln2" | "ln10" | "random" => Some(0),
        
        // Unary functions
        "sin" | "cos" | "tan" | "asin" | "acos" | "atan" |
        "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh" |
        "exp" | "exp2" | "log" | "log2" | "log10" | "sqrt" | "cbrt" |
        "floor" | "ceil" | "round" | "trunc" | "fract" | "signum" |
        "gamma" | "lgamma" | "erf" | "erfc" |
        "degrees" | "radians" |
        "is_nan" | "is_infinite" | "is_finite" | "is_normal" => Some(1),
        
        // Binary functions  
        "pow" | "atan2" | "random_range" | "hypot" | "copysign" |
        "random_normal"
        => Some(2),
        
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_trig() {
        assert!((execute_math_builtin("sin", &[0.0]).unwrap() - 0.0).abs() < 1e-10);
        assert!((execute_math_builtin("cos", &[0.0]).unwrap() - 1.0).abs() < 1e-10);
        assert!((execute_math_builtin("tan", &[0.0]).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_trig() {
        assert!((execute_math_builtin("asin", &[0.5]).unwrap() - PI/6.0).abs() < 1e-10);
        assert!((execute_math_builtin("acos", &[0.5]).unwrap() - PI/3.0).abs() < 1e-10);
        assert!((execute_math_builtin("atan", &[1.0]).unwrap() - PI/4.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic() {
        assert!((execute_math_builtin("sinh", &[0.0]).unwrap() - 0.0).abs() < 1e-10);
        assert!((execute_math_builtin("cosh", &[0.0]).unwrap() - 1.0).abs() < 1e-10);
        assert!((execute_math_builtin("tanh", &[0.0]).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_power_functions() {
        assert!((execute_math_builtin("pow", &[2.0, 3.0]).unwrap() - 8.0).abs() < 1e-10);
        assert!((execute_math_builtin("sqrt", &[4.0]).unwrap() - 2.0).abs() < 1e-10);
        assert!((execute_math_builtin("cbrt", &[8.0]).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_special_functions() {
        // Gamma(1) = 1, Gamma(2) = 1, Gamma(3) = 2, Gamma(4) = 6
        assert!((execute_math_builtin("gamma", &[1.0]).unwrap() - 1.0).abs() < 1e-10);
        assert!((execute_math_builtin("gamma", &[2.0]).unwrap() - 1.0).abs() < 1e-10);
        assert!((execute_math_builtin("gamma", &[3.0]).unwrap() - 2.0).abs() < 1e-10);
        assert!((execute_math_builtin("gamma", &[4.0]).unwrap() - 6.0).abs() < 1e-10);
        
        // erf(0) = 0, erf(∞) = 1
        assert!((execute_math_builtin("erf", &[0.0]).unwrap() - 0.0).abs() < 1e-10);
        assert!((execute_math_builtin("erf", &[10.0]).unwrap() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_constants() {
        assert!((execute_math_builtin("pi", &[]).unwrap() - PI).abs() < 1e-15);
        assert!((execute_math_builtin("e", &[]).unwrap() - E).abs() < 1e-15);
        assert!((execute_math_builtin("tau", &[]).unwrap() - 2.0 * PI).abs() < 1e-15);
    }

    #[test]
    fn test_random_functions() {
        // Test that random functions return reasonable values
        let r = execute_math_builtin("random", &[]).unwrap();
        assert!(r >= 0.0 && r < 1.0);
        
        let r_range = execute_math_builtin("random_range", &[5.0, 10.0]).unwrap();
        assert!(r_range >= 5.0 && r_range < 10.0);
        
        // Normal distribution should be somewhat close to mean
        let normal = execute_math_builtin("random_normal", &[0.0, 1.0]).unwrap();
        assert!(normal.abs() < 10.0); // Very loose bound, but should catch obvious errors
    }

    #[test]
    fn test_utility_functions() {
        assert!((execute_math_builtin("degrees", &[PI]).unwrap() - 180.0).abs() < 1e-10);
        assert!((execute_math_builtin("radians", &[180.0]).unwrap() - PI).abs() < 1e-10);
        assert!((execute_math_builtin("hypot", &[3.0, 4.0]).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_edge_cases() {
        // Test NaN cases
        assert!(execute_math_builtin("sqrt", &[-1.0]).unwrap().is_nan());
        assert!(execute_math_builtin("asin", &[2.0]).unwrap().is_nan());
        assert!(execute_math_builtin("log", &[-1.0]).unwrap().is_nan());
        
        // Test infinity
        assert!(execute_math_builtin("exp", &[1000.0]).unwrap().is_infinite());
    }
}