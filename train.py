import numpy as np
import matplotlib.pyplot as plt
from utils import load_cifar10_data
from model import NeuralNetwork, cross_entropy

def train():
    
    data_dir = './data/cifar-10-batches-py'
    classes = [0, 1, 2]  # airplane, automobile, bird
    X_train, y_train, X_test, y_test_oh, y_test_int = load_cifar10_data(data_dir, classes)
    input_dim = X_train.shape[1]
    nn = NeuralNetwork(input_dim, 128, 64, 3)
    lr = 0.01
    epochs = 50
    batch_size = 64
    loss_curve = []
    for epoch in range(epochs):
        perm = np.random.permutation(X_train.shape[0])
        X_train_shuff = X_train[perm]
        y_train_shuff = y_train[perm]
        for i in range(0, X_train.shape[0], batch_size):
            Xb = X_train_shuff[i:i+batch_size]
            yb = y_train_shuff[i:i+batch_size]
            out = nn.forward(Xb)
            loss = cross_entropy(out, yb)
            grads = nn.backward(Xb, yb)
            nn.update(grads, lr)
        # End of epoch
        out = nn.forward(X_train)
        loss = cross_entropy(out, y_train)
        loss_curve.append(loss)
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss:.4f}")
    # Savig
    np.savez('model_weights.npz', W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2, W3=nn.W3, b3=nn.b3)
    # Plotting
    plt.plot(loss_curve)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('loss_curve.png')
    plt.show()

if __name__ == '__main__':
    train()
