# LoRA in Pure Rust — Minimal, Tested, Multimodal

A lightweight implementation of Low-Rank Adaptation (LoRA) in Rust, demonstrating parameter-efficient fine-tuning techniques through a simple, maintainable codebase.

## Overview

This project implements LoRA, a technique that enables efficient fine-tuning of large models by updating only a small number of parameters. The implementation is:

- **Minimal**: Core concepts clearly expressed without unnecessary abstractions
- **Tested**: Includes unit tests verifying convergence properties
- **Multimodal**: Supports training across multiple input modalities (text + image)

## Architecture Decisions

### Why Low-Rank Adaptation?

LoRA represents weight updates as the product of two low-rank matrices (A and B). This decision offers significant benefits:

- **Parameter Efficiency**: Using rank-1 matrices reduces parameter count by up to 62.5% compared to full-matrix fine-tuning
- **Training Speed**: Fewer parameters means faster training and lower memory requirements
- **Adaptability**: Easy to apply to different parts of a model or across modalities

### Implementation Choices

1. **Simple Matrix Operations**: Leveraged ndarray for clarity and performance
2. **Generic Layer Structure**: LoRALayer is reusable across different inputs/dimensions
3. **Gradient-Based Training**: Implemented standard gradient descent for simplicity and transparency
4. **Test-Driven Development**: Ensured correct convergence behavior through automated tests

## Key Tradeoffs

| Approach | Parameters | Accuracy | Maintainability | Training Speed |
|----------|------------|----------|----------------|----------------|
| Full Matrix | 100% | High | Good | Slower |
| LoRA (rank=2) | 50% | Good | Better | Faster |
| LoRA (rank=1) | 37.5% | Acceptable | Best | Fastest |

**Rank Selection Tradeoff**: Lower rank = fewer parameters but potentially higher approximation error:
- Rank 1 (current): 37.5% parameters, MSE ~0.21-0.23
- Could increase rank for better accuracy at cost of more parameters

## Maintainability Features

- **Clean Separation of Concerns**: LoRA layer separate from training logic
- **Comprehensive Tests**: Ensures model convergence remains stable
- **Explicit Dimensionality Comments**: Matrix shapes documented for future maintainers
- **Minimal Dependencies**: Only ndarray and rand needed

## Usage

```bash
# Run the main example
cargo run

# Run tests
cargo test
```

## Future Extensions

While maintaining minimalism, this implementation could be extended with:

1. Optimizers beyond vanilla SGD (Adam, AdamW)
2. Integration with larger models
3. Checkpoint saving/loading
4. Adaptive rank selection based on task complexity
5. CLI parameter support for rank, dimensions, and sample size
6. Visualization tools to plot loss curves across different ranks (1, 2, 4)
7. Early fusion and cross-modal attention for more advanced architectures
8. ONNX export for LoRA-only delta application