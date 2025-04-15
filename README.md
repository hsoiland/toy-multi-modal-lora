I reappraoched this in pure rust as a simple way to distil my learnings from the lora-exploration project, setting a base for my continued learning in machine learning.

A toy example allows me to show understanding of the core concepts in a small, easy to consume and understandable fasion while still proving the concept. 

# LoRA in Pure Rust — Minimal, Tested, Multimodal

A lightweight implementation of Low-Rank Adaptation (LoRA) in Rust, demonstrating parameter-efficient fine-tuning techniques through a simple, maintainable codebase.

## How LoRA Training Works

Instead of fine-tuning full weight matrices \( W \), LoRA learns an efficient low-rank approximation:

\[
\Delta W = B \cdot A
\]

Where:
- \( A \in \mathbb{R}^{r \times d} \): projects input down to a smaller latent space
- \( B \in \mathbb{R}^{d \times r} \): projects back up to match the original dimensionality
- \( r \ll d \): rank is much smaller than the full weight size

In this implementation:
- The base weights (e.g. identity matrix) are **frozen**
- Only the LoRA weights **A** and **B** are trained
- The final adapted output is computed as:

\[
x_{\text{adapted}} = x \cdot A^T \cdot B^T \cdot \text{scale}
\]

Which is equivalent to applying:

\[
x \cdot (\Delta W)^T \quad \text{where} \quad \Delta W = B \cdot A
\]

This structure enables substantial reductions in trainable parameters while preserving model performance.


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
- **Comprehensive Tests**: Ensures model convergence remains stable with tests in a dedicated directory
- **Explicit Dimensionality Comments**: Matrix shapes documented for future maintainers
- **Minimal Dependencies**: Only ndarray and rand needed
- **Well-Organized Code Structure**: Separate library and test files for better maintainability

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
9. Integration with Burn, Rust's native deep learning framework, for autodifferentiation and GPU acceleration