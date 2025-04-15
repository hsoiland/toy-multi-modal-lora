use simple_rust_lora::train_multimodal_lora;

#[test]
fn test_multimodal_lora_converges() {
    // Call the training function and check the returned loss
    let final_loss = train_multimodal_lora();
    
    // Final loss should be reasonably low
    assert!(final_loss < 0.35, "Final loss should be low: got {}", final_loss);
} 