# Homework_1

The Binary classifer and the Multi-class classifier are written in 2 python3 scripts named 'Binary_classifier.py' and 'Multi-class.py' respectively. 
Programming language used: Python 3


## Usage

In the Binary classifier (5.1) script, questions (a)-(d) are written in each individual function, with the train and test part incorporated in each of them.
In the function calls section do the following: 
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

In the Multi-class classifier (5.2) script, questions (a)-(d) are written in each individual function, with the train and test part incorporated in each of them. In the function calls section do the following: 
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
