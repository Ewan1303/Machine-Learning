"""Task 1 (checker): the pen-and-paper neuron calculations, done by NumPy.

Do the task on paper first, then run this to check your answers:

    python task_01_neuron_by_hand.py
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Part (a): one neuron, three inputs
x = np.array([1.0, 2.0, -1.0])
w = np.array([0.5, -0.25, 1.0])
b = 0.5
z = w @ x + b                      # weighted sum plus bias
print("(a) z = w.x + b =", z)
print("    sigmoid(z) =", round(float(sigmoid(z)), 4))
print("    tanh(z)    =", round(float(np.tanh(z)), 4))
print("    ReLU(z)    =", max(0.0, z))

# Part (b): the decision line of a 2-input neuron with w = (2, -1), b = -1
w2, b2 = np.array([2.0, -1.0]), -1.0
print("\n(b) boundary 2*x1 - x2 - 1 = 0, i.e. x2 = 2*x1 - 1")
for p in [(1.0, 0.0), (0.0, 0.0), (1.0, 1.0)]:
    zp = w2 @ np.array(p) + b2
    side = "positive side" if zp > 0 else "negative side" if zp < 0 else "on the line"
    print(f"    point {p}: z = {zp:+.1f} -> {side}")
print("    distance of the line from the origin |b|/||w|| =", round(abs(b2) / np.linalg.norm(w2), 4))

# Part (c): scaling w and b together keeps the line but sharpens the sigmoid
p = np.array([0.6, 0.0])            # a point just on the positive side
for scale in (1, 2, 10):
    zs = scale * (w2 @ p + b2)
    print(f"(c) scale {scale:>2}: z = {zs:+.2f}, sigmoid = {sigmoid(zs):.4f}")

# Part (d): gradient of the sigmoid output with respect to w and b (example from part a)
s = sigmoid(z)
ds_dz = s * (1 - s)                # derivative of the sigmoid
print("\n(d) sigma'(z) =", round(float(ds_dz), 4))
print("    d a / d w =", ds_dz * x, "  d a / d b =", round(float(ds_dz), 4))

# Part (e): one perceptron update with a step activation, target 1, learning rate 0.5
eta, target = 0.5, 1
pred = int(z > 0)
w_new = w + eta * (target - pred) * x
b_new = b + eta * (target - pred)
z_new = w_new @ x + b_new
print(f"\n(e) prediction before = {pred}; w_new = {w_new}, b_new = {b_new}; z_new = {z_new} -> prediction {int(z_new > 0)}")
