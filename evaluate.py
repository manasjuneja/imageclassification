import numpy as np
from utils import load_cifar10_data
from model import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def evaluate():
    data_dir = './data/cifar-10-batches-py'
    classes = [0, 1, 2]
    X_train, y_train, X_test, y_test_oh, y_test_int = load_cifar10_data(data_dir, classes)
    input_dim = X_train.shape[1]
    nn = NeuralNetwork(input_dim, 128, 64, 3)
    # Load trained weights
    weights = np.load('model_weights.npz')
    nn.W1, nn.b1 = weights['W1'], weights['b1']
    nn.W2, nn.b2 = weights['W2'], weights['b2']
    nn.W3, nn.b3 = weights['W3'], weights['b3']
    # Predict
    out = nn.forward(X_test)
    y_pred = np.argmax(out, axis=1)
    y_true = y_test_int
    # Metrics
    acc = np.mean(y_pred == y_true)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    print("Confusion Matrix:\n", cm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Accuracy:", acc)
    # Plot confusion matrix
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == '__main__':
    evaluate()
