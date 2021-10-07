from utils import mnist_reader
import numpy
import torch
import numpy as np
from numpy import genfromtxt
import torchvision

#import cupy as cp

X_train, Y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, Y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
X_train = X_train/255
T = 20

f_train = open("train.txt", "w")
f_test = open("test.txt", "w")

#print(len(X_train[0]))
def predict(w, x_t):
    yhat_t = numpy.dot(w, x_t)
    yhat_label = int(numpy.sign(yhat_t))
    return yhat_label

def get_label(y_t):
    label = 0
    if y_t%2 == 0:
        label = 1
    else: 
        label = -1

    return label

def get_tau(y_t, x_t, w):
    x_t_mag = numpy.linalg.norm(x_t)
    tau = (1 - y_t*np.dot(w, x_t))/x_t_mag**2
    return tau

def train_test(X_train, Y_train, X_test, Y_test, tau, T):
    weights = numpy.zeros((1, 784))
    for i in range(T):
        ############################ Train Part ############################# 
        n_mistake_train = 0
        n_total_train = 0
        for j in range (len(X_train)):
            #print(weights.shape)
            y_t = get_label(Y_train[j])
            x_t = X_train[j]
            np.reshape(x_t, (784, 1))
            y_hat = int(np.sign(np.dot(weights, x_t)))
            n_total_train+=1
            if y_hat != y_t:                #Check Mistake
                weights += tau * y_t * x_t
                n_mistake_train+=1
        print(n_mistake_train)
        train_acc = 100*((n_total_train - n_mistake_train)/n_total_train)
        #print(100*((n_total-n_mistake)/n_total))
        f_train.write("{}\n".format(train_acc))
        optim_weights = weights
        
        ############################ Test Part #############################
        n_mistake_test = 0
        n_total_test = 0
        for j in range (len(X_test)):
            y_label = get_label(Y_test[j])
            x_test = X_test[j]
            np.reshape(x_test, (784, 1))
            y_hat = int(np.sign(np.dot(optim_weights, x_test)))
            n_total_test+=1
            if y_hat != y_label:            #Check Mistake
                n_mistake_test+=1
        print(n_mistake_test)
        test_acc = 100*((n_total_test - n_mistake_test)/n_total_test)
        f_test.write("{}\n".format(test_acc))
    #return(weights)

def PA_train_test(X_train, Y_train, X_test, Y_test, T):
    weights = numpy.zeros((1, 784))
    for i in range(T):
        ############################ Train Part ############################# 
        n_mistake_train = 0
        n_total_train = 0
        pa_tau = 0
        for j in range (len(X_train)):
            #print(weights.shape)
            y_t = get_label(Y_train[j])
            x_t = X_train[j]
            np.reshape(x_t, (784, 1))
            y_hat = int(np.sign(np.dot(weights, x_t)))
            pa_tau = get_tau(y_t, x_t, weights)
            n_total_train+=1
            if y_hat != y_t:                #Check Mistake
                weights += pa_tau * y_t * x_t
                n_mistake_train+=1
            #print(weights)
            #print(100*((n_total-n_mistake)/n_total))
        print(n_mistake_train)
        train_acc = 100*((n_total_train - n_mistake_train)/n_total_train)
        f_train.write("{}\n".format(train_acc))
        optim_weights = weights

        ############################ Test Part #############################
        n_mistake_test = 0
        n_total_test = 0
        for j in range (len(X_test)):
            y_label = get_label(Y_test[j])
            x_test = X_test[j]
            np.reshape(x_test, (784, 1))
            y_hat = int(np.sign(np.dot(optim_weights, x_test)))
            n_total_test+=1
            if y_hat != y_label:            #Check Mistake
                n_mistake_test+=1
        print(n_mistake_test)
        test_acc = 100*((n_total_test - n_mistake_test)/n_total_test)
        f_test.write("{}\n".format(test_acc))
    #print(pa_tau)
    #return(weights)



############################ Q5.1c ############################# 
def Avg_train_test(X_train, Y_train, X_test, Y_test, tau, T):
    weights = numpy.zeros((1, 784))
    cached_weights = numpy.zeros((1, 784))
    c = 1
    for i in range(T):
        ############################ Train Part ############################# 
        n_mistake_train = 0
        n_total_train = 0
        for j in range (len(X_train)):
            #print(weights.shape)
            y_t = get_label(Y_train[j])
            x_t = X_train[j]
            np.reshape(x_t, (784, 1))
            y_hat = int(np.sign(np.dot(weights, x_t)))
            n_total_train+=1
            if y_hat != y_t:                #Check Mistake
                weights += tau * y_t * x_t  #weight Update
                cached_weights += tau * y_t * x_t * c   #Cached weights update
                n_mistake_train+=1
            c+=1
        print(n_mistake_train)
        train_acc = 100*((n_total_train - n_mistake_train)/n_total_train)
        #print(100*((n_total-n_mistake)/n_total))
        f_train.write("{}\n".format(train_acc))
        optim_weights = weights - (cached_weights * 1/c)
        
        ############################ Test Part #############################
        n_mistake_test = 0
        n_total_test = 0
        for j in range (len(X_test)):
            y_label = get_label(Y_test[j])
            x_test = X_test[j]
            np.reshape(x_test, (784, 1))
            y_hat = int(np.sign(np.dot(optim_weights, x_test)))
            n_total_test+=1
            if y_hat != y_label:            #Check Mistake
                n_mistake_test+=1
        print(n_mistake_test)
        test_acc = 100*((n_total_test - n_mistake_test)/n_total_test)
        f_test.write("{}\n".format(test_acc))
    #return(weights)

def Avg_PA_train_test(X_train, Y_train, X_test, Y_test, T):
    weights = numpy.zeros((1, 784))
    cached_weights = numpy.zeros((1, 784))
    c = 1
    for i in range(T):
        ############################ Train Part ############################# 
        n_mistake_train = 0
        n_total_train = 0
        for j in range (len(X_train)):
            #print(weights.shape)
            y_t = get_label(Y_train[j])
            x_t = X_train[j]
            np.reshape(x_t, (784, 1))
            y_hat = int(np.sign(np.dot(weights, x_t)))
            pa_tau = get_tau(y_t, x_t, weights)     #Calculate tau
            n_total_train+=1
            if y_hat != y_t:                #Check Mistake
                weights += pa_tau * y_t * x_t  #weight Update
                cached_weights += pa_tau * y_t * x_t * c   #Cached weigths update
                n_mistake_train+=1
            c+=1
        print(n_mistake_train)
        train_acc = 100*((n_total_train - n_mistake_train)/n_total_train)
        #print(100*((n_total-n_mistake)/n_total))
        f_train.write("{}\n".format(train_acc))
        optim_weights = weights - (cached_weights * 1/c)
        
        ############################ Test Part #############################
        n_mistake_test = 0
        n_total_test = 0
        for j in range (len(X_test)):
            y_label = get_label(Y_test[j])
            x_test = X_test[j]
            np.reshape(x_test, (784, 1))
            y_hat = int(np.sign(np.dot(optim_weights, x_test)))
            n_total_test+=1
            if y_hat != y_label:            #Check Mistake
                n_mistake_test+=1
        print(n_mistake_test)
        test_acc = 100*((n_total_test - n_mistake_test)/n_total_test)
        f_test.write("{}\n".format(test_acc))

def General_Learning(X_train, Y_train, X_test, Y_test, tau, T):
    weights = numpy.zeros((1, 784))
    for i in range(T):
        ############################ Train Part ############################# 
        n_mistake_train = 0
        n_total_train = 0
        train_data = 100
        for j in range (train_data):
            #print(weights.shape)
            y_t = get_label(Y_train[j])
            x_t = X_train[j]
            np.reshape(x_t, (784, 1))
            y_hat = int(np.sign(np.dot(weights, x_t)))
            n_total_train+=1
            if y_hat != y_t:                #Check Mistake
                weights += tau * y_t * x_t
                n_mistake_train+=1
        print(n_mistake_train)
        train_acc = 100*((n_total_train - n_mistake_train)/n_total_train)
        #print(100*((n_total-n_mistake)/n_total))
        f_train.write("{}\n".format(train_acc))
        optim_weights = weights
        train_data+=100
        
        ############################ Test Part #############################
        n_mistake_test = 0
        n_total_test = 0
        for j in range (len(X_test)):
            y_label = get_label(Y_test[j])
            x_test = X_test[j]
            np.reshape(x_test, (784, 1))
            y_hat = int(np.sign(np.dot(optim_weights, x_test)))
            n_total_test+=1
            if y_hat != y_label:            #Check Mistake
                n_mistake_test+=1
        print(n_mistake_test)
        test_acc = 100*((n_total_test - n_mistake_test)/n_total_test)
        f_test.write("{}\n".format(test_acc))

############################ Function calls #############################

#train_test(X_train, Y_train, X_test, Y_test, 1, T)     #Un-comment and run script for Standard perceptron (Q5.1a&b)
#PA_train_test(X_train, Y_train, X_test, Y_test, T)      #Un-comment and run script for Passive-Aggresive Algorithm (Q5.1a&b)
#Avg_train_test(X_train, Y_train, X_test, Y_test, 1, T)      #Un-comment and run script for Averaged Perceptron (Q5.1c)
#Avg_PA_train_test(X_train, Y_train, X_test, Y_test, T)      #Un-comment and run script for Average Passive-Aggresive Perceptron (Q5.1c)
General_Learning(X_train, Y_train, X_test, Y_test, 1, T)     #Un-comment and run script for Standard perceptron (Q5.1d)

