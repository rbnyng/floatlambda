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

pub fn execute_math_builtin(op: &str, args: &[f64]) -> Result<f64, EvalError> {
    match op {
        // Basic transcendental functions
        "sin" => Ok(args[0].sin()),
        "cos" => Ok(args[0].cos()),
        "tan" => Ok(args[0].tan()),
        "exp" => Ok(args[0].exp()),
        "log" => {
            if args[0] <= 0.0 {
                Ok(f64::NAN)
            } else {
                Ok(args[0].ln())
            }
        }
        
        // Inverse trig functions
        "asin" => {
            if args[0] < -1.0 || args[0] > 1.0 {
                Ok(f64::NAN)
            } else {
                Ok(args[0].asin())
            }
        }
        "acos" => {
            if args[0] < -1.0 || args[0] > 1.0 {
                Ok(f64::NAN)
            } else {
                Ok(args[0].acos())
            }
        }
        "atan" => Ok(args[0].atan()),
        "atan2" => Ok(args[0].atan2(args[1])), // atan2(y, x)
        
        // Hyperbolic functions
        "sinh" => Ok(args[0].sinh()),
        "cosh" => Ok(args[0].cosh()),
        "tanh" => Ok(args[0].tanh()),
        "asinh" => Ok(args[0].asinh()),
        "acosh" => {
            if args[0] < 1.0 {
                Ok(f64::NAN)
            } else {
                Ok(args[0].acosh())
            }
        }
        "atanh" => {
            if args[0] <= -1.0 || args[0] >= 1.0 {
                Ok(f64::NAN)
            } else {
                Ok(args[0].atanh())
            }
        }
        
        // Power and root functions
        "pow" => Ok(args[0].powf(args[1])),
        "sqrt" => {
            if args[0] < 0.0 {
                Ok(f64::NAN)
            } else {
                Ok(args[0].sqrt())
            }
        }
        "cbrt" => Ok(args[0].cbrt()), // Cube root
        "exp2" => Ok(args[0].exp2()), // 2^x
        "log2" => {
            if args[0] <= 0.0 {
                Ok(f64::NAN)
            } else {
                Ok(args[0].log2())
            }
        }
        "log10" => {
            if args[0] <= 0.0 {
                Ok(f64::NAN)
            } else {
                Ok(args[0].log10())
            }
        }
        
        // Rounding and utility functions
        "floor" => Ok(args[0].floor()),
        "ceil" => Ok(args[0].ceil()),
        "round" => Ok(args[0].round()),
        "trunc" => Ok(args[0].trunc()),
        "fract" => Ok(args[0].fract()),
        "signum" => Ok(args[0].signum()),
        
        // Special functions
        "gamma" => {
            if args[0] <= 0.0 && args[0].fract() == 0.0 {
                // Gamma is undefined for non-positive integers
                Ok(f64::NAN)
            } else {
                Ok(gamma_lanczos(args[0]))
            }
        }
        "lgamma" => {
            if args[0] <= 0.0 && args[0].fract() == 0.0 {
                Ok(f64::NAN)
            } else {
                Ok(gamma_lanczos(args[0]).ln())
            }
        }
        "erf" => Ok(erf_approx(args[0])),
        "erfc" => Ok(1.0 - erf_approx(args[0])),
        
        // Random number generation
        "random" => {
            let mut rng = rand::thread_rng();
            Ok(rng.gen::<f64>()) // Random float in [0, 1)
        }
        "random_range" => {
            let mut rng = rand::thread_rng();
            Ok(rng.gen_range(args[0]..args[1]))
        }
        "random_normal" => {
            // Box-Muller transform for normal distribution
            let mut rng = rand::thread_rng();
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            Ok(args[0] + args[1] * z0) // mean + std * z0
        }
        
        // Mathematical constants
        "pi" => Ok(PI),
        "e" => Ok(E),
        "tau" => Ok(TAU),
        "sqrt2" => Ok(SQRT_2),
        "ln2" => Ok(LN_2),
        "ln10" => Ok(LN_10),
        
        // Additional utility functions
        "degrees" => Ok(args[0] * 180.0 / PI), // Radians to degrees
        "radians" => Ok(args[0] * PI / 180.0), // Degrees to radians
        "hypot" => Ok(args[0].hypot(args[1])), // sqrt(x² + y²)
        "copysign" => Ok(args[0].abs() * args[1].signum()),
        
        // Floating point utilities
        "is_nan" => Ok(if args[0].is_nan() { 1.0 } else { 0.0 }),
        "is_infinite" => Ok(if args[0].is_infinite() { 1.0 } else { 0.0 }),
        "is_finite" => Ok(if args[0].is_finite() { 1.0 } else { 0.0 }),
        "is_normal" => Ok(if args[0].is_normal() { 1.0 } else { 0.0 }),
        
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
        "pow" | "atan2" | "random_range" | "hypot" | "copysign" => Some(2),
        
        // Ternary functions
        "random_normal" => Some(2), // mean, std
        
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