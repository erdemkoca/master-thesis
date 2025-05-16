## Summary: LassoNet - Neural Networks with Feature Sparsity

**Reference**: Lemhadri et al., 2021

### Key Idea
LassoNet extends the idea of Lasso to neural networks by enforcing global feature sparsity. It forces features to influence the network only if they also contribute linearly to the output, combining feature selection and deep learning in one model.



- LassoNet extends the Lasso (L1-regularized regression) principle to **neural networks**.
- It enforces **global feature sparsity** across the entire network using a hierarchical structure.
- Features can only influence hidden layers if they also influence the skip (input-output) layer.

### Architecture
- **Input → Skip Connection → Output** (linear path)
- **Input → Neural Network → Output** (nonlinear path)
- **Feature sparsity is enforced only via skip-connection weights**.

### Optimization
- Uses a **Hierarchical Proximal Operator (Hier-Prox)** to efficiently handle the sparsity constraint.
- Trains models from **dense to sparse** (warm-start optimization).

### Performance
- LassoNet achieves superior performance compared to traditional feature selection methods (e.g., Fisher Score, HSIC-Lasso).
- Shows robustness on real-world datasets (MNIST, ISOLET, Mice Proteins).
- Retains high accuracy even with very few selected features.

### Extensions
- LassoNet can be applied to:
  - **Unsupervised learning** (autoencoders)
  - **Matrix completion** (missing data problems)
- Future work proposes adapting LassoNet to convolutional neural networks (CNNs) to achieve **filter sparsity**.

### Key Advantages
- Globally consistent feature sparsity.
- Smooth trade-off between pure Lasso and standard Neural Networks.
- More interpretable and efficient models.

### Possible Discussion Points
- How LassoNet balances accuracy and interpretability.
- Application to medical datasets (e.g., Framingham Heart Study).
- Handling class imbalance and highly correlated features.
- Computational cost compared to simpler feature selection methods.

---
