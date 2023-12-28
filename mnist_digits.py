import mnist_loader
import numpy as np
import matplotlib.pyplot as plt

training_data, _, _ = mnist_loader.load_data_wrapper()
training_data = list(training_data)
print(training_data[0][0])

n = 0
image_grid = training_data[n][0].reshape(28,28)
digit_vector = training_data[n][1]
digit = np.argmax(digit_vector)

plt.imshow(image_grid, cmap="gray")
plt.title(f"Correct Digit: {digit}")
plt.axis("off")
plt.show()