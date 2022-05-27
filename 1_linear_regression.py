import numpy as np 
import torch

#Training data
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

#Weights and biases
w = torch.randn(2,3,requires_grad=True)
b = torch.randn(2, requires_grad=True)

def model(x):
    return x @ w.t() + b

#Loss mean squared error
def mse(t1,t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


#Train for 100 epochs
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    #Compute gradients
    loss.backward()
    #Update the weights and biases
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

preds = model(inputs)
loss = mse(preds, targets)
print(loss)