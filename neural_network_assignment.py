import numpy as np
import matplotlib.pyplot as plt

# ─── 1. Activation Functions ───────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# ─── 2. Sample Data ────────────────────────────────────────────────────────
x1 = 3
x2 = 2
y_actual = 17

# ─── 3. Initialize Weights (random between 1 and 10) ──────────────────────
np.random.seed(42)
w1 = float(np.random.randint(1, 11))
w2 = float(np.random.randint(1, 11))
print(f"Initial weights:  w1 = {w1},  w2 = {w2}")

# ─── Hyperparameter ────────────────────────────────────────────────────────
learning_rate = 0.01
epochs = 20

# ─── History for plotting ──────────────────────────────────────────────────
error_history = []
ypred_history = []

# ─── 4-6. Training Loop (Feed Forward → Error → Backprop) ─────────────────
for epoch in range(1, epochs + 1):

    # Feed Forward
    y_pred = x1 * w1 + x2 * w2

    # Error Calculation
    error = (y_actual - y_pred) ** 2

    # Back Propagation
    dE_dw1 = -2 * (y_actual - y_pred) * x1
    dE_dw2 = -2 * (y_actual - y_pred) * x2

    # Update weights
    w1 -= learning_rate * dE_dw1
    w2 -= learning_rate * dE_dw2

    error_history.append(error)
    ypred_history.append(y_pred)

    print(f"Epoch {epoch:2d} | y_pred = {y_pred:8.4f} | error = {error:10.4f} | w1 = {w1:.4f} | w2 = {w2:.4f}")

print(f"\nFinal weights:  w1 = {w1:.4f},  w2 = {w2:.4f}")
print(f"Final y_pred   = {y_pred:.4f}  (target = {y_actual})")

# ─── 7. Visualization ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Neural Network Training Progress", fontsize=14)

axes[0].plot(range(1, epochs + 1), error_history, color="crimson", marker="o", markersize=4)
axes[0].set_title("Squared Error vs Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Squared Error")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(range(1, epochs + 1), ypred_history, color="steelblue", marker="o", markersize=4, label="y_pred")
axes[1].axhline(y=y_actual, color="green", linestyle="--", linewidth=1.5, label=f"y_actual = {y_actual}")
axes[1].set_title("Predicted Value vs Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("y_pred")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/training_plots.png", dpi=150)
print("Plot saved.")
