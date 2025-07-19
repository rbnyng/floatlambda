// src/ml/optimizers.rs

use crate::error::EvalError;
use super::tensor::DifferentiableTensor;

// Performs a single step of Stochastic Gradient Descent.
// new_params = params - learning_rate * gradients
// This function creates a new tensor for the updated parameters.
pub fn sgd_update(
    params: &DifferentiableTensor,
    grads: &DifferentiableTensor,
    learning_rate: f64,
) -> Result<DifferentiableTensor, EvalError> {
    if params.shape != grads.shape {
        return Err(EvalError::TypeError(format!(
            "Shape mismatch for sgd_update: params shape {:?} vs grads shape {:?}",
            params.shape, grads.shape
        )));
    }

    let new_data: Vec<f64> = params
        .data
        .iter()
        .zip(grads.data.iter())
        .map(|(p, g)| p - learning_rate * g)
        .collect();

    // The resulting tensor is a new leaf node in the graph; it has no context.
    let updated_params = DifferentiableTensor::new(params.shape.clone(), new_data);
    Ok(updated_params)
}

// Creates the initial state [m, v, t] for the AdamW optimizer.
// m and v are zero-tensors with the same shape as the parameters.
// t is the initial timestep, 0.0.
pub fn adamw_init_state(
    params: &DifferentiableTensor,
) -> (DifferentiableTensor, DifferentiableTensor, f64) {
    let zeros = vec![0.0; params.data.len()];
    let m = DifferentiableTensor::new(params.shape.clone(), zeros.clone());
    let v = DifferentiableTensor::new(params.shape.clone(), zeros);
    let t = 0.0;
    (m, v, t)
}

// Performs a single step of the AdamW optimizer.
pub fn adamw_update(
    params: &DifferentiableTensor,
    grads: &DifferentiableTensor,
    m: &DifferentiableTensor,
    v: &DifferentiableTensor,
    t: f64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
) -> Result<(DifferentiableTensor, DifferentiableTensor, DifferentiableTensor, f64), EvalError> {
    if params.shape != grads.shape || params.shape != m.shape || params.shape != v.shape {
        return Err(EvalError::TypeError(
            "Shape mismatch between params, grads, and optimizer state tensors.".to_string(),
        ));
    }

    let t_new = t + 1.0;

    // Update biased first moment estimate
    let m_new_data: Vec<f64> = m.data.iter()
        .zip(grads.data.iter())
        .map(|(m_i, g_i)| beta1 * m_i + (1.0 - beta1) * g_i)
        .collect();

    // Update biased second raw moment estimate
    let v_new_data: Vec<f64> = v.data.iter()
        .zip(grads.data.iter())
        .map(|(v_i, g_i)| beta2 * v_i + (1.0 - beta2) * (g_i * g_i))
        .collect();

    // Compute bias-corrected first moment estimate
    let m_hat_data: Vec<f64> = m_new_data.iter()
        .map(|&m_i| m_i / (1.0 - beta1.powf(t_new)))
        .collect();

    // Compute bias-corrected second raw moment estimate
    let v_hat_data: Vec<f64> = v_new_data.iter()
        .map(|&v_i| v_i / (1.0 - beta2.powf(t_new)))
        .collect();
    
    // Decoupled weight decay
    let params_decayed_data: Vec<f64> = params.data.iter()
        .map(|&p_i| p_i * (1.0 - lr * weight_decay))
        .collect();

    // Update parameters
    let params_new_data: Vec<f64> = params_decayed_data.iter()
        .zip(m_hat_data.iter())
        .zip(v_hat_data.iter())
        .map(|((p_i, m_hat_i), v_hat_i)| p_i - lr * m_hat_i / (v_hat_i.sqrt() + epsilon))
        .collect();

    // Create new tensors for the results
    let new_params = DifferentiableTensor::new(params.shape.clone(), params_new_data);
    let new_m = DifferentiableTensor::new(m.shape.clone(), m_new_data);
    let new_v = DifferentiableTensor::new(v.shape.clone(), v_new_data);

    Ok((new_params, new_m, new_v, t_new))
}
