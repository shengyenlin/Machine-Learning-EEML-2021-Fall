import numpy as np
import pandas as pd
import math

class LogisticRegression:
    def __init__(self):
        self.train_valid_ratio = 0.7
        self.train_acc_list = list()
        self.train_loss_list = list()
        self.valid_acc_list = list()
        self.valid_loss_list = list()
        #Best weights
        self.best_w = None
        self.best_b = None
        #Best results
        self.best_epoch = None
        self.best_valid_loss = None
        self.best_valid_acc = None

    def initialize_params(self, x):
        w = np.random.rand(x.shape[1])
        bias = np.random.rand()
        return w, bias

    def train(self, X, y, batch_size, epoch_size, learning_rate, verbose = True):
        w, b = self.initialize_params(X)
        #adagrad params
        eps = 1e-12
        g_b = 0
        g_w = np.ones(X.shape[1])

        #other hyperparams
        best_valid_loss = 99999
        patience = 10 #for early stopping
        
        for num_epoch in range(1, epoch_size+1):
            #Shuffle when each epoch begin
            index = np.arange(X.shape[0])
            np.random.shuffle(index)
            X = X[index]
            y = y[index]
            split_point_x = math.floor(X.shape[0] * self.train_valid_ratio)
            split_point_y = math.floor(y.shape[0] * self.train_valid_ratio)
            X_train = X[:split_point_x, :]
            y_train = y[:split_point_y]
            X_valid = X[split_point_x:, :] 
            y_valid = y[split_point_y:]

            for num_batch in range(int(X_train.shape[0] / batch_size)):
                #print("start")
                x_batch = X_train[num_batch * batch_size:(num_batch + 1) * batch_size]
                y_batch = y_train[num_batch * batch_size:(num_batch + 1) * batch_size]

                #implement adagrad
                w_grad, b_grad = self.compute_gradient(x_batch, y_batch, w, b)
                g_w += w_grad ** 2
                g_b += b_grad ** 2

                w = w - learning_rate * w_grad / np.sqrt(g_w + eps)
                b = b - learning_rate * b_grad / np.sqrt(g_b + eps)
                
            #compute loss 
            y_train_pred = np.round(self.compute_logistic_value(X_train, w, b))
            train_acc = self.compute_accuracy(y_train_pred, y_train)
            train_loss = self.compute_cross_entropy_loss(y_train_pred, y_train) / X_train.shape[0]
            self.train_acc_list.append(train_acc)
            self.train_loss_list.append(train_loss)

            y_valid_pred = np.round(self.compute_logistic_value(X_valid, w, b))
            valid_acc = self.compute_accuracy(y_valid_pred, y_valid)
            valid_loss = self.compute_cross_entropy_loss(y_valid_pred, y_valid) / X_valid.shape[0]
            self.valid_acc_list.append(valid_acc)
            self.valid_loss_list.append(valid_loss)

            if verbose:
                print(f"Epoch {num_epoch}, train loss = {round(train_loss, 4)} (Accuracy: {round(train_acc*100, 3)}%), valid loss = {round(valid_loss, 4)} (Accuracy: {round(valid_acc*100, 3)}%)")
        
            #save best result
            if valid_loss < best_valid_loss:
                self.best_w = w
                self.best_b = b
                self.best_epoch = num_epoch
                best_valid_loss = valid_loss
                best_valid_acc = valid_acc
                self.best_valid_loss = best_valid_loss
                self.best_valid_acc = best_valid_acc

            #early stopping
            if valid_loss > best_valid_loss and num_epoch >= self.best_epoch + patience:
                self.stop_epoch = self.best_epoch + patience
                if verbose:
                    print("Early Stopping!")
                    print("="*10 + "validation result" + "="*10)
                    print(f"Best epoch is {self.best_epoch} with minimum valid loss = {round(best_valid_loss, 4)} (Accuracy: {round(best_valid_acc*100, 3)}%)")
                return

        self.stop_epoch = num_epoch
        if verbose:
            print("Finish model tuning")
            print("="*10 + "Model result" + "="*10)
            print(f"Best epoch is {self.best_epoch} with minimum valid loss = {round(best_valid_loss, 4)} (Accuracy: {round(best_valid_acc*100, 3)}%)")

    def train_with_full_data(self, X, y, batch_size, epoch_size, learning_rate, verbose = False):
        w, b = self.initialize_params(X)
        #adagrad params
        eps = 1e-12
        g_b = 0
        g_w = np.ones(X.shape[1])
        
        for num_epoch in range(1, epoch_size+1):
            #Shuffle when each epoch begin
            index = np.arange(X.shape[0])
            np.random.shuffle(index)
            X = X[index]
            y = y[index]

            for num_batch in range(int(X.shape[0] / batch_size)):
                x_batch = X[num_batch * batch_size:(num_batch + 1) * batch_size]
                y_batch = y[num_batch * batch_size:(num_batch + 1) * batch_size]

                #implement adagrad
                w_grad, b_grad = self.compute_gradient(x_batch, y_batch, w, b)
                g_w += w_grad ** 2
                g_b += b_grad ** 2

                w = w - learning_rate * w_grad / np.sqrt(g_w + eps)
                b = b - learning_rate * b_grad / np.sqrt(g_b + eps)
                
            #compute loss 
            y_pred = np.round(self.compute_logistic_value(X, w, b))
            train_acc = self.compute_accuracy(y_pred, y)
            train_loss = self.compute_cross_entropy_loss(y_pred, y) / X.shape[0]

            if verbose:
                print(f"Epoch {num_epoch}, train loss = {round(train_loss, 4)} (Accuracy: {round(train_acc*100, 3)}%)")

        self.best_w = w
        self.best_b = b
        return

    def predict(self, X_test):
        y_pred = self.compute_logistic_value(X_test, self.best_w, self.best_b)
        y_pred = np.round(y_pred)
        return y_pred

    def compute_gradient(self, X, y_true, w, b):
        #print(w.shape)
        y_pred = self.compute_logistic_value(X, w, b).flatten() #dim = (batch_size, )
        pred_error = y_true - y_pred
        w_grad = -np.dot(X.T, pred_error) #dim = (feature_size, )
        b_grad = -pred_error.sum(axis = 0)
        return w_grad, b_grad

    def compute_logistic_value(self, X, w, b):
        return self.sigmoid(np.matmul(X, w) + b)

    def compute_cross_entropy_loss(self, y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1-eps)
        cross_entropy = -np.dot(y_true, np.log(y_pred )) - np.dot((1-y_true), np.log(1 - y_pred))
        return cross_entropy

    def compute_accuracy(self, y_pred, y_true):
        accuracy = 1 - np.mean(np.abs(y_pred - y_true))
        return accuracy

    def sigmoid(self, z):
        res = 1 / (1.0 + np.exp(-z))
        return np.clip(res, 1e-6, 1 - (1e-6))