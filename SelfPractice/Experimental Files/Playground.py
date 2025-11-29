import numpy as np

sizes = [2, 3, 1]  # Example sizes, you can change these values
weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
print(weights[1])
print(weights)

