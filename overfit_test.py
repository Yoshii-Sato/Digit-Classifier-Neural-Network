import matplotlib.pyplot as plt
import mnist_loader
import dc_nn_2 as nn

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = nn.Network([784, 30, 10])
epoch_list, training_accuracy, training_cost, test_accuracy, test_cost = net.SGD(training_data, 30, 10, 3.0, test_data, monitor_training_accuracy=True, monitor_test_accuracy=True)

print(epoch_list)
print(training_accuracy)
print(training_cost)
print(test_accuracy)
print(test_cost)

# Graph of Epoch vs Accuracy on training data
plt.plot(epoch_list, training_accuracy)
plt.ylabel("Accuracy on training data")
plt.xlabel("Epoch")
plt.grid(True)
plt.show()

# Graph of Epoch vs Cost of training data
# plt.plot(epoch_list, training_cost)
# plt.ylabel("Cost of training data")
# plt.xlabel("Epoch")
# plt.grid(True)
# plt.show()

# Graph of Epoch vs Accuracy on test data
plt.plot(epoch_list, test_accuracy)
plt.ylabel("Accuracy on test data")
plt.xlabel("Epoch")
plt.grid(True)
plt.show()

# Graph of Epoch vs Cost on test data
# plt.plot(epoch_list, test_cost)
# plt.ylabel("Accuracy of test data")
# plt.xlabel("Epoch")
# plt.grid(True)
# plt.show()