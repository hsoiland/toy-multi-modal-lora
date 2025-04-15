use ndarray::{Array2, Array};
use rand::prelude::*;

pub struct LoRALayer {
    pub rank: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub a: Array2<f32>,
    pub b: Array2<f32>,
    pub scale: f32,
}

impl LoRALayer {
    pub fn new(in_dim: usize, out_dim: usize, rank: usize, scale: f32) -> Self {
        // Initialize with small random values to break symmetry
        let mut rng = rand::thread_rng();
        let a = Array::from_shape_fn((rank, in_dim), |_| rng.gen_range(-0.1..0.1));
        let b = Array::from_shape_fn((out_dim, rank), |_| rng.gen_range(-0.1..0.1));
        Self { rank, in_dim, out_dim, a, b, scale }
    }

    pub fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
        let temp = x.dot(&self.a.t());
        let out = temp.dot(&self.b.t());
        out * self.scale
    }
}

pub fn generate_multimodal_data(samples: usize, dim_text: usize, dim_image: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let mut rng = rand::thread_rng();
    let text = Array::from_shape_fn((samples, dim_text), |_| rng.gen_range(-1.0..1.0));
    let image = Array::from_shape_fn((samples, dim_image), |_| rng.gen_range(-1.0..1.0));
    let fusion = ndarray::concatenate![ndarray::Axis(1), text.clone(), image.clone()];
    let true_w = Array::eye(dim_text + dim_image);
    let noise = Array::from_shape_fn((samples, dim_text + dim_image), |_| rng.gen_range(-0.1f32..0.1f32));
    let y = fusion.dot(&true_w.t()) + 0.05 * noise;
    (text, image, y)
}

pub fn train_multimodal_lora() -> f32 {
    let dim_text = 4;
    let dim_image = 4;
    let dim_out = dim_text + dim_image;
    let rank = 1;
    let scale = 1.0;  // Using 1.0 for better learning
    let epochs = 500;
    let lr = 0.01f32;  // Explicit f32 type for learning rate
    let samples = 100;

    let (x_text, x_image, y) = generate_multimodal_data(samples, dim_text, dim_image);
    let mut lora_text = LoRALayer::new(dim_text, dim_out, rank, scale);
    let mut lora_image = LoRALayer::new(dim_image, dim_out, rank, scale);

    println!("Starting multimodal training with rank={}", rank);
    println!("Text input shape: {:?}, Image input shape: {:?}", x_text.shape(), x_image.shape());
    println!("Output shape: {:?}", y.shape());
    println!("Total parameters: {} LoRA vs {} full ({:.2}%)",
             lora_text.a.len() + lora_text.b.len() + lora_image.a.len() + lora_image.b.len(),
             dim_out * (dim_text + dim_image),
             ((lora_text.a.len() + lora_text.b.len() + lora_image.a.len() + lora_image.b.len()) as f32 
              / (dim_out * (dim_text + dim_image)) as f32) * 100.0);

    for epoch in 0..epochs {
        // Forward pass
        let preds = lora_text.apply(&x_text) + lora_image.apply(&x_image);
        let error = &preds - &y;
        let loss = error.mapv(|e| e.powi(2)).mean().unwrap();
        
        // Compute gradients
        let grad_output = 2.0 * &error / (samples as f32);
        
        // Text modality gradients
        let temp_text_a = x_text.dot(&lora_text.a.t());  // samples × rank
        let grad_text_b = grad_output.t().dot(&temp_text_a);  // out_dim × rank
        
        let temp_text_b = grad_output.dot(&lora_text.b);  // samples × rank
        let grad_text_a = temp_text_b.t().dot(&x_text);  // rank × text_dim
        
        // Image modality gradients
        let temp_img_a = x_image.dot(&lora_image.a.t());  // samples × rank
        let grad_img_b = grad_output.t().dot(&temp_img_a);  // out_dim × rank
        
        let temp_img_b = grad_output.dot(&lora_image.b);  // samples × rank
        let grad_img_a = temp_img_b.t().dot(&x_image);  // rank × img_dim
        
        // Update parameters
        lora_text.a = &lora_text.a - &(lr * grad_text_a);
        lora_text.b = &lora_text.b - &(lr * grad_text_b);
        
        lora_image.a = &lora_image.a - &(lr * grad_img_a);
        lora_image.b = &lora_image.b - &(lr * grad_img_b);

        if epoch % 50 == 0 || epoch == epochs - 1 {
            println!("Epoch {epoch:>3} | Loss: {loss:.6}");
        }
    }

    println!("\nMultimodal training complete.");
    let base_output = ndarray::concatenate![ndarray::Axis(1), x_text.clone(), x_image.clone()];
    let lora_output = lora_text.apply(&x_text) + lora_image.apply(&x_image);
    let diff = &lora_output - &base_output;
    let mse = diff.mapv(|e| e.powi(2)).mean().unwrap();
    println!("MSE between multimodal LoRA output and ground truth: {:.6}", mse);

    let full_params = dim_out * (dim_text + dim_image);
    let lora_params = lora_text.a.len() + lora_text.b.len() + lora_image.a.len() + lora_image.b.len();
    let percent = (lora_params as f32 / full_params as f32) * 100.0;
    println!("\nTrained {} LoRA params vs {} full params ({:.2}%)", lora_params, full_params, percent);
    
    // Return the final MSE for testing
    mse
} 