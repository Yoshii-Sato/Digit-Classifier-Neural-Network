import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
    

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a)+b)
        return a
    

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None, monitor_training_cost=False, monitor_training_accuracy=False, monitor_test_cost=False, monitor_test_accuracy=False):
        epoch_list = []
        training_cost = []
        training_accuracy = []
        test_cost = []
        test_accuracy = []
        
        training_data = list(training_data)
        training_data_len = len(training_data)

        if(test_data):
            test_data = list(test_data)
            test_data_len = len(test_data)

        for epoch in range(epochs):
            epoch_list.append(epoch) #monitoring changes in each epoch
            np.random.shuffle(training_data)
            mini_batches = [training_data[i: i+mini_batch_size] for i in range(0, training_data_len, mini_batch_size)]
            for mini_batch in mini_batches: 
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print("Epoch {}: {} / {}".format(epoch, self.evaluate(test_data), test_data_len))
            else:
                print("Epoch {} complete!".format(epoch))

            # Monitors the changes in cost/accuracy of both training and testing data
            if monitor_training_accuracy:
                accuracy = self.evaluate(training_data) / training_data_len
                training_accuracy.append(accuracy)

            if monitor_training_cost:
                pass

            if monitor_test_accuracy:
                accuracy = self.evaluate(test_data) / test_data_len
                test_accuracy.append(accuracy)

            if monitor_test_cost:
                pass
            
        return epoch_list, training_accuracy, training_cost, test_accuracy, test_cost
            
    def update_mini_batch(self, mini_batch, eta):   
        total_nabla_weight = [np.zeros(w.shape) for w in self.weights]
        total_nabla_bias = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            delta_nabla_weight, delta_nabla_bias = self.backprop(x, y)

            total_nabla_weight = [w + dw for w, dw in zip(total_nabla_weight, delta_nabla_weight)]
            total_nabla_bias = [b + db for b, db in zip(total_nabla_bias, delta_nabla_bias)]

        self.weights = [w - (eta / len(mini_batch))*dw for w, dw in zip(self.weights, total_nabla_weight)]
        self.biases = [b - (eta / len(mini_batch))*db for b, db in zip(self.biases, total_nabla_bias)]

    
    def backprop(self, x, y):
        nabla_weight = [np.zeros(w.shape) for w in self.weights]
        nabla_bias = [np.zeros(b.shape) for b in self.biases]

        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_weight[-1] = np.dot(delta, activations[-2].transpose())
        nabla_bias[-1] = delta

        for layer in range(2, self.num_layers):
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sigmoid_prime(zs[-layer])
            nabla_weight[-layer] = np.dot(delta, activations[-layer-1].transpose())
            nabla_bias[-layer] = delta

        return (nabla_weight, nabla_bias)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)

    def cost(self, test_data):
        pass

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
    

def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
        return sigmoid(z)*(1-sigmoid(z))

