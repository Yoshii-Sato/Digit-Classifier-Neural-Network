import matplotlib.pyplot as plt
import mnist_loader
import digit_classifier_nn as nn

epoch_list = []
accuracy_list = []

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = nn.Network([784, 30, 10])
net.SGD(training_data, 5, 10, 3.0, test_data, epoch_list, accuracy_list)

print(epoch_list)
print(accuracy_list)

# Graph of Epoch vs Accuracy on training data
plt.plot(epoch_list, accuracy_list)
plt.ylabel("Accuracy on training data")
plt.xlabel("Epoch")
plt.grid(True)
plt.show()

# Graph of Epoch vs Cost on training data
# Graph of Epoch vs Cost on test data
# Graph of Epoch vs Accuracy on training data