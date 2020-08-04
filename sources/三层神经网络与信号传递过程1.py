X = np.array([1.0, 0.5]) 
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) 
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape) # (2, 3)
print(X.shape)  # (2,)
print(B1.shape) # (3,)

A1 = np.dot(X, W1) + B1