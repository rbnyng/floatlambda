#[cfg(test)]
mod cnn_test {

    use float_lambda::memory::{Heap, HeapObject};
    use float_lambda::ml::{
        autodiff, ops, optimizers,
        tensor::DifferentiableTensor,
    };
    use rand::{thread_rng, Rng};

    // Helper to create a tensor with random data and register it on the heap.
    fn init_tensor(heap: &mut Heap, shape: Vec<usize>) -> u64 {
        let num_elements = shape.iter().product();
        let mut rng = thread_rng();
        let data: Vec<f64> = (0..num_elements).map(|_| rng.gen_range(-0.1..0.1)).collect();
        let tensor = DifferentiableTensor::new(shape, data);
        heap.register(HeapObject::Tensor(tensor))
    }

    // A struct to hold the heap IDs of our model's parameters.
    struct SimpleCNN {
        conv_w: u64,
        conv_b: u64,
        dense_w: u64,
        dense_b: u64,
    }

    impl SimpleCNN {
        fn new(heap: &mut Heap) -> Self {
            Self {
                conv_w: init_tensor(heap, vec![3, 3, 1, 2]), // [h, w, in_c, out_c]
                conv_b: init_tensor(heap, vec![2]),
                dense_w: init_tensor(heap, vec![8, 2]),
                dense_b: init_tensor(heap, vec![1, 2]), // Shape should be [1, 2] to match matmul output
            }
        }

        // Helper to get a list of all parameter IDs.
        fn params_list(&self) -> Vec<u64> {
            vec![self.conv_w, self.conv_b, self.dense_w, self.dense_b]
        }
    }

    // The forward pass implemented by calling Rust functions directly.
    fn forward_pass(heap: &mut Heap, model: &SimpleCNN, inputs_id: u64) -> u64 {
        let conv_w = heap.get_tensor_mut(model.conv_w).unwrap().clone();
        let conv_b = heap.get_tensor_mut(model.conv_b).unwrap().clone();
        let dense_w = heap.get_tensor_mut(model.dense_w).unwrap().clone();
        let dense_b = heap.get_tensor_mut(model.dense_b).unwrap().clone();
        let inputs = heap.get_tensor_mut(inputs_id).unwrap().clone();

        // Layer 1: Conv
        let conv1_res = ops::conv2d(inputs_id, &inputs, model.conv_w, &conv_w, model.conv_b, &conv_b, 1, 1).unwrap();
        let conv1_id = heap.register(HeapObject::Tensor(conv1_res));
        let conv1 = heap.get_tensor_mut(conv1_id).unwrap().clone();

        // Layer 2: ReLU
        let relu1_res = ops::relu(conv1_id, &conv1);
        let relu1_id = heap.register(HeapObject::Tensor(relu1_res));
        let relu1 = heap.get_tensor_mut(relu1_id).unwrap().clone();

        // Layer 3: Pool
        let pool1_res = ops::max_pool2d(relu1_id, &relu1, 2, 2).unwrap();
        let pool1_id = heap.register(HeapObject::Tensor(pool1_res));
        let pool1 = heap.get_tensor_mut(pool1_id).unwrap().clone();

        // Layer 4: Flatten
        let flat_res = ops::flatten(pool1_id, &pool1).unwrap();
        let flat_id = heap.register(HeapObject::Tensor(flat_res));
        let flat = heap.get_tensor_mut(flat_id).unwrap().clone();

        // Layer 5: Dense
        let matmul_res = ops::matmul(flat_id, &flat, model.dense_w, &dense_w).unwrap();
        let matmul_id = heap.register(HeapObject::Tensor(matmul_res));
        let matmul = heap.get_tensor_mut(matmul_id).unwrap().clone();

        let logits_res = ops::add(matmul_id, &matmul, model.dense_b, &dense_b).unwrap();
        heap.register(HeapObject::Tensor(logits_res))
    }

    #[test]
    #[ignore]
    fn cnn_training_pipeline_loss_decreases() {
        let mut heap = Heap::new();

        // 1. Initialize model and optimizer state
        let model = SimpleCNN::new(&mut heap);
        let mut opt_states: Vec<_> = model.params_list().into_iter().map(|p_id| {
            let params = heap.get_tensor_mut(p_id).unwrap().clone();
            let (m, v, t) = optimizers::adamw_init_state(&params);
            (heap.register(HeapObject::Tensor(m)), heap.register(HeapObject::Tensor(v)), t)
        }).collect();

        // 2. Create fake data
        let inputs_id = init_tensor(&mut heap, vec![1, 5, 5, 1]); // [N, H, W, C]
        let labels = DifferentiableTensor::new(vec![1, 2], vec![0.0, 1.0]); // One-hot
        let labels_id = heap.register(HeapObject::Tensor(labels));

        // 3. Calculate initial loss
        let initial_logits_id = forward_pass(&mut heap, &model, inputs_id);
        let initial_labels = heap.get_tensor_mut(labels_id).unwrap().clone();
        let initial_logits = heap.get_tensor_mut(initial_logits_id).unwrap().clone();
        let initial_loss_res = ops::softmax_ce_loss(labels_id, &initial_labels, initial_logits_id, &initial_logits).unwrap();
        let initial_loss = initial_loss_res.data[0];
        println!("Initial Loss: {}", initial_loss);

        let mut current_params = model.params_list();
        let num_steps = 5;

        // 4. Run training loop
        for i in 0..num_steps {
            // --- Forward Pass & Loss ---
            let logits_id = forward_pass(&mut heap, &SimpleCNN {
                conv_w: current_params[0], conv_b: current_params[1],
                dense_w: current_params[2], dense_b: current_params[3],
            }, inputs_id);
            let current_labels = heap.get_tensor_mut(labels_id).unwrap().clone();
            let current_logits = heap.get_tensor_mut(logits_id).unwrap().clone();
            let loss_res = ops::softmax_ce_loss(labels_id, &current_labels, logits_id, &current_logits).unwrap();
            let loss_id = heap.register(HeapObject::Tensor(loss_res));

            // --- Backward Pass ---
            heap.get_tensor_mut(loss_id).unwrap().grad.borrow_mut()[0] = 1.0;
            let topo_order = autodiff::build_topo_order(loss_id, &heap);
            for &node_id in topo_order.iter().rev() {
                let (context, grad_data) = {
                    let tensor = heap.get_tensor_mut(node_id).unwrap();
                    (tensor.context.clone(), tensor.grad.borrow().clone())
                };
                if let Some(ctx) = context {
                    (ctx.backward_fn)(&grad_data, &mut heap);
                }
            }

            // --- Optimizer Step ---
            let mut new_params = Vec::new();
            for (j, &p_id) in current_params.iter().enumerate() {
                let params = heap.get_tensor_mut(p_id).unwrap().clone();
                let grads = DifferentiableTensor::new(params.shape.clone(), params.grad.borrow().clone());
                let (m_id, v_id, t) = opt_states[j];
                let m = heap.get_tensor_mut(m_id).unwrap().clone();
                let v = heap.get_tensor_mut(v_id).unwrap().clone();

                let (new_p, new_m, new_v, new_t) = optimizers::adamw_update(&params, &grads, &m, &v, t, 0.01, 0.9, 0.999, 1e-8, 0.01).unwrap();
                
                new_params.push(heap.register(HeapObject::Tensor(new_p)));
                opt_states[j] = (heap.register(HeapObject::Tensor(new_m)), heap.register(HeapObject::Tensor(new_v)), new_t);
            }
            current_params = new_params;
            
            println!("Step {}: Loss: {}", i + 1, heap.get_tensor_mut(loss_id).unwrap().data[0]);
        }

        // 5. Calculate final loss and assert
        let final_logits_id = forward_pass(&mut heap, &SimpleCNN {
            conv_w: current_params[0], conv_b: current_params[1],
            dense_w: current_params[2], dense_b: current_params[3],
        }, inputs_id);
        let final_labels = heap.get_tensor_mut(labels_id).unwrap().clone();
        let final_logits = heap.get_tensor_mut(final_logits_id).unwrap().clone();
        let final_loss_res = ops::softmax_ce_loss(labels_id, &final_labels, final_logits_id, &final_logits).unwrap();
        let final_loss = final_loss_res.data[0];
        println!("Final Loss: {}", final_loss);

        assert!(final_loss < initial_loss, "FAIL: Loss did not decrease after training.");
        println!("\n--- PASS: Loss decreased after training. ---");
    }

}
