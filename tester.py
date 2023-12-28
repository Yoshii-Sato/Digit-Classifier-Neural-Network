import digit_classifier_nn as nn
import mnist_loader
import matplotlib.pyplot as plt
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = nn.Network([784, 30, 10])
net.SGD(training_data, 10, 10, 3.0, test_data)

training_data_2, validation_data_2, test_data_2 = mnist_loader.load_data_wrapper()
training_data_2_list = list(training_data_2)
output = net.feedforward(training_data_2_list[0][0])
formatted_output = np.vectorize(lambda x: format(x, 'f'))(output)
print(formatted_output)


image_grid = training_data_2_list[0][0].reshape(28,28)
digit_vector = training_data_2_list[0][1]
digit = np.argmax(digit_vector)

plt.imshow(image_grid, cmap="gray") 
plt.title(f"Digit: {digit}")
plt.axis("off")
plt.show()
