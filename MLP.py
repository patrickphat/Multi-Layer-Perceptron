import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_MOMENTUM = 0.9

class TwoLayerPerceptron:
    
    def __init__(self,hidden_units,initializer="xavier"):
        initializers = {"xavier":self.init_params_xavier,
                        "normal": self.init_params_normal}

        # Number
        self.hidden_units = hidden_units
        
        # Number of classes
        self.num_classes = None
              
        # Number of training data
        self.num_data = None 

        # Get weights initializer
        self.initializer=initializers[initializer]

        # Store weights for the model
        self.parameters = {"W1":None,
                           "b1":None,
                           "W2":None,
                           "b2":None}
        
        # Store cache to compute back propagation
        self.cache = {"A1":None,
                      "A2":None,
                      "W1":None,
                      "W2":None,
                      "b1":None,
                      "b2":None}
        
        # Save history
        self.history = {"train_acc":[],
                        "train_loss":[],
                        "val_acc":[],
                        "val_loss":[]}
  
    def relu(self,Z):

        # Times Z with the mask of where Z > 0
        A = (Z>0)*Z
      
        return A

    def softmax(self,Z):

        A = np.exp(Z)/np.sum(np.exp(Z),axis=0)
        return A

    def init_params_xavier(self,n_in,n_out):
        sigma = 2/np.sqrt(n_in+n_out)
        W = np.random.normal(loc=0,scale=sigma,size=(n_out,n_in))*0.01
        b = np.random.normal(loc=0,scale=sigma,size=(n_out,1))*0.01
        return W,b
    
    def init_params_normal(self,n_in,n_out):
        W = np.random.normal(size=(n_out,n_in))*0.01
        b = np.random.normal(size=(n_out,1))*0.01
        return W,b

    def categorical_cross_entropy(self,y_true,y_pred): 
 
         # sparsing y_true
        y_true_ = OneHotEncoder().fit_transform(y_true.reshape(-1,1)).todense()
        # print("y_true",y_true.shape)
        # print("y_true_",y_true_.shape)
        # print("y_pred {} ,y_true_ {}".format(y_pred.shape,y_true_.shape))

        # # Add small number to resolve log 0 
        y_pred += 1e-5


        # Compute cross entropy
        CE =  (np.multiply(-y_true_,np.log(y_pred))).mean()
        return CE

    def forward(self,X,y, mode = "train" ):

        # Retrieve weights from the model
        W1,b1 = self.parameters["W1"], self.parameters["b1"]
        W2,b2 = self.parameters["W2"], self.parameters["b2"]

        # Feed foward
        Z1 = W1.dot(X.T)
        A1 = self.relu(Z1)
        Z2 = W2.dot(Z1)
        A2 = self.softmax(Z2)
        y_hat = A2.argmax(axis=0)

        # Save cache for backpropagation
        self.cache["A1"] = A1
        self.cache["A2"] = A2

        # Compute loss
        loss = self.categorical_cross_entropy(y,A2.T)
        accuracy = accuracy_score(y,y_hat)

        if mode == "train":
          self.history["train_loss"].append(loss)
          self.history["train_acc"].append(accuracy)
        if mode == "val":
          self.history["val_loss"].append(loss)
          self.history["val_acc"].append(accuracy)

        return accuracy,loss
        
    def backward(self,X,y,learning_rate = DEFAULT_LEARNING_RATE,momentum=DEFAULT_MOMENTUM): 

        # Notations:
        # n_h: number hidden unit
        # n_o: number output unit
        # n_i: input shape 

        # Retrieve number of data
        m = self.num_data

        # Retrieve cache to perform backprop
        A1 = self.cache["A1"] # shape (n_h,m)
        A2 = self.cache["A2"] # shape (n_o,m)
        
        # Retrieve weights to perform backprop
        W2 = self.parameters["W2"]

        # Compute derivative respected to loss function
        y_true = OneHotEncoder().fit_transform(y.reshape(-1,1)).todense()
        dZ2 = 1/m*(A2-y_true.T) # shape (n_o,m)
        dW2 = dZ2.dot(A1.T) # shape (n_o,n_h)
        db2  = dZ2.sum(axis=1)

        relu_mask_A1 = (A1<0) # shape(n_h,m), only where > 0 got update
        dZ1 = W2.T.dot(dZ2) # shape(n_h,m)
        dZ1[relu_mask_A1] = 0 
        dW1 = dZ1.dot(X) # shape (n_h,n_i)
        db1  = dZ1.sum(axis=1) # shape (n_h,1)
        
        # Retrieve old weights 
        W1_old = self.cache["W1"]
        b1_old = self.cache["b1"]
        W2_old = self.cache["W2"]
        b2_old = self.cache["b2"]

        # Update new old weights for calculating momentum
        self.cache["W1"] = self.parameters["W1"].copy()
        self.cache["b1"] = self.parameters["b1"].copy()
        self.cache["W2"] = self.parameters["W2"].copy()
        self.cache["b2"] = self.parameters["b2"].copy()

        # If input a list of learning rates
        if type(learning_rate) == list:
          self.parameters["W1"] -= learning_rate[0]*dW1 - momentum*(self.parameters["W1"]-W1_old)
          self.parameters["b1"] -= learning_rate[0]*db1 - momentum*(self.parameters["b1"]-b1_old)
          self.parameters["W2"] -= learning_rate[1]*dW2 - momentum*(self.parameters["W2"]-W2_old)
          self.parameters["b2"] -= learning_rate[1]*db2 - momentum*(self.parameters["b2"]-b2_old)
        else:
          # Update weights
          self.parameters["W1"] -= learning_rate*dW1 - momentum*(self.parameters["W1"]-W1_old)
          self.parameters["b1"] -= learning_rate*db1 - momentum*(self.parameters["b1"]-b1_old)
          self.parameters["W2"] -= learning_rate*dW2 - momentum*(self.parameters["W2"]-W2_old)
          self.parameters["b2"] -= learning_rate*db2 - momentum*(self.parameters["b2"]-b2_old)

    def step(self,X,y, learning_rate = DEFAULT_LEARNING_RATE,momentum=DEFAULT_MOMENTUM):
        
        accuracy,loss = self.forward(X,y,mode="train")
        self.backward(X,y,learning_rate=learning_rate,momentum=momentum)
        
        # self.history["train_acc"]
        return accuracy,loss

    def fit(self,X,y, epochs=10,learning_rate = DEFAULT_LEARNING_RATE,
            momentum = DEFAULT_MOMENTUM, validation_split = 0.2,print_every=100):

        # Get number of classes by getting number of unique value in y
        self.num_classes = len(np.unique(y))

        # Get number of data
        self.num_data = X.shape[0]

        # Get input shape
        self.input_units = X.shape[1]
        
        # Initialize weight
        self.parameters["W1"],self.parameters["b1"] = self.initializer(n_in=self.input_units,n_out=self.hidden_units)       
        self.parameters["W2"],self.parameters["b2"] = self.initializer(n_in=self.hidden_units,n_out=self.num_classes)

        # Initialize cache weights for momentum
        self.cache["W1"], self.cache["b1"] = self.parameters["W1"].copy(),self.parameters["b1"].copy()
        self.cache["W2"], self.cache["b2"] = self.parameters["W2"].copy(),self.parameters["b2"].copy()
        
        # Split for validation 
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=validation_split)

        for i in range(epochs):
          train_acc,train_loss = self.step(x_train,y_train,
                                           learning_rate=learning_rate,momentum=momentum)
          val_acc,val_loss = self.forward(x_val,y_val,mode="val")

          if not (i+1)%print_every:
            print("epoch {}: train_acc {} train_loss {} val_acc {} val_loss {}".format(i+1,
                                                                                     train_acc,train_loss,
                                                                                     val_acc,val_loss))
            
    def evaluate(self,x_test,y_test):
        test_acc,test_loss = self.forward(x_test,y_test)
        print("test_acc {} test_loss {}".format(test_acc,test_loss))
        return test_acc,test_loss

    def predict (self,X):
        
        # Retrieve weights from the model
        W1,b1 = self.parameters["W1"], self.parameters["b1"]
        W2,b2 = self.parameters["W2"], self.parameters["b2"]

        # Feed foward
        Z1 = W1.dot(X.T)
        A1 = self.relu(Z1)
        Z2 = W2.dot(Z1)
        A2 = self.softmax(Z2)
        y_hat = A2.argmax(axis=0)

        return y_hat
    
