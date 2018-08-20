import numpy as np
from perceptron import Perceptron

# defining a
a = [-1, -1,  1, -1,  1,
     -1,  1, -1,  1, -1,
     -1,  1,  1,  1, -1,
     1, -1, -1, -1,  1,
     1, -1, -1, -1,  1]
# defining A inverted
ai = [1, -1, -1, -1,  1,
      1, -1, -1, -1,  1,
      -1,  1,  1,  1, -1,
      -1,  1, -1,  1, -1,
      -1, -1,  1, -1, -1]

if __name__ == '__main__':
    neural_network = Perceptron(5 * 5)
    train_cases = [neural_network.generate_data(a, 1)
                   for _ in range(10)]
    train_labels = [-1 for _ in range(10)]
    aux1 = [neural_network.generate_data(ai, 1)
            for _ in range(10)]
    aux2 = [1 for _ in range(10)]
    train_cases = train_cases + aux1
    train_labels = train_labels + aux2
    test_cases = [neural_network.generate_data(a, 1)
                  for _ in range(5)]
    test_labels = [-1 for _ in range(5)]
    aux1 = [neural_network.generate_data(ai, 1)
            for _ in range(5)]
    aux2 = [1 for _ in range(5)]
    test_cases = test_cases + aux1
    test_labels = test_labels + aux2
    neural_network.train(train_cases, train_labels, 100, 0.01)
    outs = [neural_network.recognize(x) for x in test_cases]
    accuracy = sum([1 if y == label else 0
                    for y, label in zip(outs, test_labels)])
    print('accuracy:', accuracy/len(outs))
