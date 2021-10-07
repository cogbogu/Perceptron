# Perceptron

The Binary classifer and the Multi-class classifier are written in 2 python3 scripts named 'Binary_classifier.py' and 'Multi-class.py' respectively. 
Programming language used: Python 3


## Usage

In the Binary classifier script, I wrote functions that implement Standard and Average perceptron with and without a passive-aggressive algorithm, and a general learning paradigm for Binary classification of the Fashion-MNIST dataset into odd and even labels. To make the function calls in the Binary_classifier.py script, do the following: 
```python
train_test(X_train, Y_train, X_test, Y_test, 1, T)     #Un-comment and run script for Standard perceptron (Q5.1a&b)

PA_train_test(X_train, Y_train, X_test, Y_test, T)      #Un-comment and run script for Passive-Aggresive Algorithm (Q5.1a&b)

Avg_train_test(X_train, Y_train, X_test, Y_test, 1, T)      #Un-comment and run script for Averaged Perceptron (Q5.1c)

Avg_PA_train_test(X_train, Y_train, X_test, Y_test, T)      #Un-comment and run script for Average Passive-Aggresive Perceptron (Q5.1c)

General_Learning(X_train, Y_train, X_test, Y_test, 1, T)     #Un-comment and run script to evaluate General Learning Curve (Q5.1d)
```
To run the Binary classification script, in terminal use: 
```bash
python Binary_classifier.py
```

Similarly, in the Multi-class classifier script, I wrote functions that implement Standard and Average perceptron with and without a passive-aggressive algorithm, and a general learning paradigm for multi-classification of the Fashion-MNIST dataset into 10 labels. To make the function calls in the Multi-class.py script, do the following: 
```python
train_test(X_train, Y_train, 1, T)        #Un-comment and run script for Standard perceptron (Q5.2a&b)

pa_train(X_train, Y_train, T)             #Un-comment and run script for Passive-Aggresive Algorithm (Q5.2a&b)

Avg_Perceptron(X_train, Y_train, 1, T)    #Un-comment and run script for Averaged Perceptron (Q5.2c)

Avg_PA_Perceptron(X_train, Y_train, T)    #Un-comment and run script for Average Passive-Aggresive Perceptron (Q5.2c)

General_Learning(X_train, Y_train, 1, T)  #Un-comment and run script to evaluate General Learning Curve (Q5.2d)
```

To run the Multi-class classification script, in terminal use: 
```bash
python Multi-class.py
```

## Program Output

The train.txt and test.txt file contain the train and test accuracy for each run of the experiment. 


## License
[MIT](https://choosealicense.com/licenses/mit/)
