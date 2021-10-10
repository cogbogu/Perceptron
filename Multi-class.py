from utils import mnist_reader
import numpy
import numpy as np

#import cupy as cp

X_train, Y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, Y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
X_train = X_train/255
n_classes = 10
T = 20

f_train = open("train.txt", "w")
f_test = open("test.txt", "w")


def aug_Feat(x_t, y):
    F = np.zeros((10, 784))
    F[y,:] = x_t
    return F
    
def get_tau(w, F_diff):
    weights = w.flatten()
    F_d = F_diff.flatten()
    np.reshape(weights, (7840, 1))
    num = 1 - np.dot(F_d, weights)
    denm = (numpy.linalg.norm(F_d))**2
    tau = num/denm
    return tau

def predict(w, x_t):
    y_vec = np.zeros(10)
    for i in range(len(w)):
        np.reshape(w[i], (784, 1))
        y_vec[i] = np.dot(x_t, w[i])
    y_hat = np.argmax(y_vec)
    return y_hat

def train_test(X_train, Y_train, tau, T):
    ############################ Train Part #############################
    weights = numpy.zeros((10, 784))
    for i in range(T):
        n_mistake = 0
        n_total = 0
        for j in range (len(X_train)):
            #print(weights.shape)
            x_t = X_train[j]
            y_t = Y_train[j]
            y_hat = predict(weights, x_t)
            F = aug_Feat(x_t, y_t)
            F_hat = aug_Feat(x_t, y_hat)
            n_total+=1
            if y_t != y_hat:
                #print((F - F_hat))
                weights += tau * (F - F_hat)
                #print((F - F_hat))
                n_mistake+=1
        print(n_mistake)
        train_acc = 100*((n_total - n_mistake)/n_total)
        optim_weights = weights             #optimized weights
        f_train.write("{}\n".format(train_acc))
        #print(100*((n_total-n_mistake)/n_total))
        ############################ Test Part #############################
        n_mistake_test = 0
        n_total_test = 0
        for j in range (len(X_test)):
            y_label = Y_test[j]
            x_test = X_test[j]
            np.reshape(x_test, (784, 1))
            y_hat = predict(optim_weights, x_test)
            n_total_test+=1
            if y_hat != y_label:            #Check Mistake
                n_mistake_test+=1
        print(n_mistake_test)
        test_acc = 100*((n_total_test - n_mistake_test)/n_total_test)
        #f_test.write("{}\n".format(test_acc))
        f_test.write("{}\n".format(test_acc))

def pa_train(X_train, Y_train, T):
    weights = numpy.zeros((10, 784))
    for i in range(T):
        ############################ Train Part #############################
        n_mistake = 0
        n_total = 0
        for j in range (len(X_train)):
            #print(weights.shape)
            x_t = X_train[j]
            y_t = Y_train[j]
            y_hat = predict(weights, x_t)
            F = aug_Feat(x_t, y_t)
            F_hat = aug_Feat(x_t, y_hat)
            #print(tau)
            n_total+=1
            if y_t != y_hat:
                tau = get_tau(weights, (F - F_hat))
                #print((F - F_hat))
                weights += tau * (F - F_hat)
                
                n_mistake+=1
        print(n_mistake)
        #print(100*((n_total-n_mistake)/n_total))
        train_acc = 100*((n_total - n_mistake)/n_total)
        optim_weights = weights             #optimized weights
        f_train.write("{}\n".format(train_acc))
        #print(100*((n_total-n_mistake)/n_total))
        ############################ Test Part #############################
        n_mistake_test = 0
        n_total_test = 0
        for j in range (len(X_test)):
            y_label = Y_test[j]
            x_test = X_test[j]
            np.reshape(x_test, (784, 1))
            y_hat = predict(optim_weights, x_test)
            n_total_test+=1
            if y_hat != y_label:            #Check Mistake
                n_mistake_test+=1
        print(n_mistake_test)
        test_acc = 100*((n_total_test - n_mistake_test)/n_total_test)
        #f_test.write("{}\n".format(test_acc))
        f_test.write("{}\n".format(test_acc))

def Avg_Perceptron(X_train, Y_train, tau, T):
    ############################ Train Part #############################
    weights = numpy.zeros((10, 784))
    cached_weights = numpy.zeros((10, 784))
    c = 1
    for i in range(T):
        n_mistake = 0
        n_total = 0
        for j in range (len(X_train)):
            #print(weights.shape)
            x_t = X_train[j]
            y_t = Y_train[j]
            y_hat = predict(weights, x_t)
            F = aug_Feat(x_t, y_t)
            F_hat = aug_Feat(x_t, y_hat)
            n_total+=1
            if y_t != y_hat:
                #print((F - F_hat))
                weights += tau * (F - F_hat)
                cached_weights += tau * (F - F_hat) * c
                #print((F - F_hat))
                n_mistake+=1
            c+=1
        print(n_mistake)
        train_acc = 100*((n_total - n_mistake)/n_total)
        optim_weights = weights - (cached_weights * 1/c)             #optimized weights
        f_train.write("{}\n".format(train_acc))
        #print(100*((n_total-n_mistake)/n_total))
        ############################ Test Part #############################
        n_mistake_test = 0
        n_total_test = 0
        for j in range (len(X_test)):
            y_label = Y_test[j]
            x_test = X_test[j]
            np.reshape(x_test, (784, 1))
            y_hat = predict(optim_weights, x_test)
            n_total_test+=1
            if y_hat != y_label:            #Check Mistake
                n_mistake_test+=1
        print(n_mistake_test)
        test_acc = 100*((n_total_test - n_mistake_test)/n_total_test)
        #f_test.write("{}\n".format(test_acc))
        f_test.write("{}\n".format(test_acc))

def Avg_PA_Perceptron(X_train, Y_train, T):
    ############################ Train Part #############################
    weights = numpy.zeros((10, 784))
    cached_weights = numpy.zeros((10, 784))
    c = 1
    for i in range(T):
        n_mistake = 0
        n_total = 0
        for j in range (len(X_train)):
            #print(weights.shape)
            x_t = X_train[j]
            y_t = Y_train[j]
            y_hat = predict(weights, x_t)
            F = aug_Feat(x_t, y_t)
            F_hat = aug_Feat(x_t, y_hat)
            n_total+=1
            if y_t != y_hat:
                #print((F - F_hat))
                tau = get_tau(weights, (F - F_hat))
                weights += tau * (F - F_hat)
                cached_weights += tau * (F - F_hat) * c
                #print((F - F_hat))
                n_mistake+=1
            c+=1
        print(n_mistake)
        train_acc = 100*((n_total - n_mistake)/n_total)
        optim_weights = weights - (cached_weights * 1/c)             #optimized weights
        f_train.write("{}\n".format(train_acc))
        #print(100*((n_total-n_mistake)/n_total))
        ############################ Test Part #############################
        n_mistake_test = 0
        n_total_test = 0
        for j in range (len(X_test)):
            y_label = Y_test[j]
            x_test = X_test[j]
            np.reshape(x_test, (784, 1))
            y_hat = predict(optim_weights, x_test)
            n_total_test+=1
            if y_hat != y_label:            #Check Mistake
                n_mistake_test+=1
        print(n_mistake_test)
        test_acc = 100*((n_total_test - n_mistake_test)/n_total_test)
        #f_test.write("{}\n".format(test_acc))
        f_test.write("{}\n".format(test_acc))

def General_Learning(X_train, Y_train, tau, T):
    ############################ Train Part #############################
    weights = numpy.zeros((10, 784))
    for i in range(T):
        n_mistake = 0
        n_total = 0
        train_data = 100
        for j in range (train_data):
            #print(weights.shape)
            x_t = X_train[j]
            y_t = Y_train[j]
            y_hat = predict(weights, x_t)
            F = aug_Feat(x_t, y_t)
            F_hat = aug_Feat(x_t, y_hat)
            n_total+=1
            if y_t != y_hat:
                #print((F - F_hat))
                weights += tau * (F - F_hat)
                #print((F - F_hat))
                n_mistake+=1
        print(n_mistake)
        train_acc = 100*((n_total - n_mistake)/n_total)
        optim_weights = weights             #optimized weights
        f_train.write("{}\n".format(train_acc))
        #print(100*((n_total-n_mistake)/n_total))
        train_data+=100
        ############################ Test Part #############################
        n_mistake_test = 0
        n_total_test = 0
        for j in range (len(X_test)):
            y_label = Y_test[j]
            x_test = X_test[j]
            np.reshape(x_test, (784, 1))
            y_hat = predict(optim_weights, x_test)
            n_total_test+=1
            if y_hat != y_label:            #Check Mistake
                n_mistake_test+=1
        print(n_mistake_test)
        test_acc = 100*((n_total_test - n_mistake_test)/n_total_test)
        #f_test.write("{}\n".format(test_acc))
        f_test.write("{}\n".format(test_acc))


############################ Function calls #############################

#train_test(X_train, Y_train, 1, T)
#pa_train(X_train, Y_train, T)
#Avg_Perceptron(X_train, Y_train, 1, T)
#Avg_PA_Perceptron(X_train, Y_train, T)
General_Learning(X_train, Y_train, 1, T)

