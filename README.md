# Inverse Kinematics Neural Network Using Machine Learning

## Description

This project implements a neural network to solve the inverse kinematics problem for a two-joint planar manipulator. The neural network is trained to predict the joint angles required to achieve a desired end-effector position.

## Table of Contents  
- [Installation](#installation)  
- [Usage](#usage)
  - [Initialize the Arm Object](#initializethearmobject)
  - [Train the Neural Network](#traintheneuralnetwork)
  - [Plot the Arm Configuration](#plotthearmconfiguration)
- [Contributing](#contributing)  
<a name="installation"/>


## Installation

To run this project, you need to have Python and the following libraries installed:
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

```bash
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
```
### Initialize the Arm Object:
<a name="initializethearmobject"/>

```bash
class arm():
    ### the arm class contains all the methods for defining a two joints planar manipulator,
    ### and implement a neural network inverse kinematics solver for it

    def __init__(self, links = [10, 10], origin = [0, 0], init = [0, 0]):
        # class constructor, defining the basic attributes of the arm and initial configuration
        self.link1 = links[0]
        self.link2 = links[1]
        self.x0 = origin[0]
        self.y0 = origin[1]
        self.joint1 = init[0]
        self.joint2 = init[1]
        self.direct_kin()

    def direct_kin(self):
        # this forward kinematic function calculates the Cartesian coordinates for the current joint configuration
        [self.X, self.Y] = direct_kin_([self.joint1, self.joint2], [self.link1, self.link2], [self.x0, self.y0])

    def plot_arm(self):
        # 2D plot of the current arm configuration
        plt.plot([-20,20],[0,0],'k')
        plt.plot(self.X, self.Y, linewidth=2.0)
```
### Train the Neural Network:
<a name="traintheneuralnetwork"/>

```bash
# Define hyperparameters and train the neural network
# Example hyperparameters
hidden_layers = (100, 50)
activation = 'relu'
learning_rate = 'adaptive'
solver = 'sgd'
max_iter = 500

# Generate some example training data
X_train = np.random.rand(1000, 2)  # Example input data
y_train = np.random.rand(1000, 2)  # Example output data
X_test = np.random.rand(200, 2)    # Example test data
y_test = np.random.rand(200, 2)    # Example test labels

# Initialize and train the neural network
nn = MLPRegressor(hidden_layer_sizes=hidden_layers, activation=activation, learning_rate=learning_rate, solver=solver, max_iter=max_iter)
nn.fit(X_train, y_train)

# Evaluate the neural network
train_accuracy = nn.score(X_train, y_train)
test_accuracy = nn.score(X_test, y_test)
print(f'Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')
```
### Plot the arm configuration
<a name="plotthearmconfiguration"/>

```bash
a = arm()
a.plot_arm()
plt.show()
```

## Contributing
`<a name="contributing"/>`

If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue to discuss what you would like to change.
