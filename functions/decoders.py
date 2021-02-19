"""
List of decoding algorithms

reference: https://github.com/KordingLab/Neural_Decoding/tree/master/Neural_Decoding
"""

# import required packages
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from numpy.linalg import inv, pinv
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from .generator import data_generator
from .preprocess import flatten_list
from .metrics import compute_rmse

def rmse(y_true, y_pred):
    """
    Define root mean squared error (RMSE) loss function

    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

class prediction_history(Callback):
    """
    Callback for printing validation loss after each epoch
    """
    def __init__(self, X_valid, Y_valid):
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.i_epoch = 0
        self.pred_hist = []
    def on_epoch_end(self, epoch, logs={}):
        Y_valid_predict = self.model.predict(self.X_valid)
        rmse_valid = compute_rmse(self.Y_valid,Y_valid_predict)
        self.pred_hist.append(np.mean(rmse_valid))
        self.i_epoch += 1
        print("Epoch %d >> valid loss: %.4f"%(self.i_epoch,np.mean(rmse_valid)))    
        
class lstm_decoder:
    def __init__(self,num_layers=1,units=100,batch_size=32,epochs=5,dropout=0,stateful=False,shuffle=True,verbose=0):
        self.num_layers = num_layers
        self.units = units
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.stateful = stateful
        self.shuffle = shuffle
        self.verbose = verbose
    
    def compile(self, **params):
        # params = {'units':100, 'num_epochs':5, batch_size':32, 'dropout':0., 'optimizer':'RMSprop', 'lrate':0.,}
        self.num_layers = params['num_layers']
        self.units = params['units']
        self.batch_size = params['batch_size']   
        self.timesteps = params['timesteps']
        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        self.optimizer = params['optimizer']
        self.seed = params['seed']
        
        if self.optimizer=='RMSprop':
            optim = RMSprop(lr=params['lrate'])
        else:
            optim = Adam(lr=params['lrate'])
        
        # Create new model or load existing model
        if params['load']:
            print("Loading existing model " +params['load_name'])
            model = load_model(params['load_name'],custom_objects={'rmse':rmse})
        else:
            model = Sequential()
            if self.num_layers > 1:
                for i in range(self.num_layers-1):
                    model.add(LSTM(self.units, input_shape=(self.timesteps,self.input_dim), dropout=self.dropout, stateful=self.stateful, return_sequences=True))
                model.add(LSTM(self.units,input_shape=(self.timesteps,self.input_dim), dropout=self.dropout, stateful=self.stateful))
            else:
                model.add(LSTM(self.units, input_shape=(self.timesteps,self.input_dim), dropout=self.dropout, stateful=self.stateful))
            if self.dropout>0.: 
                model.add(Dropout(self.dropout))    
            model.add(Dense(self.output_dim))
            # Compile model
            model.compile(optimizer=optim,loss=rmse) #Set loss function and optimizer
            
        #print(model.summary())
        # Print parameter count
        num_params = model.count_params()
        print('# network parameters: ' + str(num_params))
        self.model = model
        return model
        
    def fit(self,X_train,Y_train,X_valid,Y_valid,**params):
        self.epochs = params['epochs']
        self.dropout = params['dropout']
        self.stateful = params['stateful']
        self.shuffle = params['shuffle']
        self.verbose = params['verbose']
        self.fit_gen = params['fit_gen']
        # Fit model
        loss_train = []
        loss_valid = []
        predict_history = prediction_history(X_valid,Y_valid)
        if params['retrain']:
            if self.stateful:
                if self.fit_gen:
                    train_generator = data_generator(X_train,Y_train,self.batch_size,self.shuffle)
                    for i in range(self.epochs):                        
                        hist = self.model.fit_generator(generator=train_generator,
                                                        validation_data=(X_valid,Y_valid),
                                                        epochs=1,shuffle=self.shuffle,verbose=self.verbose,
                                                        callbacks=[predict_history])
                        loss_train.append(hist.history['loss'])
                        loss_valid.append(hist.history['val_loss'])
                        self.model.reset_states()
                else:
                    for i in range(self.epochs):
                        hist = self.model.fit(X_train,Y_train,
                                              validation_data=(X_valid,Y_valid),
                                              batch_size=self.batch_size,epochs=1,verbose=self.verbose,shuffle=self.shuffle,
                                              callbacks=[predict_history])
                        loss_train.append(hist.history['loss'])
                        loss_valid.append(hist.history['val_loss'])
                        self.model.reset_states()
            else:
                hist = self.model.fit(X_train,Y_train,
                                 validation_data=(X_valid,Y_valid),
                                 batch_size=self.batch_size,epochs=self.epochs,verbose=self.verbose,shuffle=self.shuffle,
                                 callbacks=[predict_history])
                loss_train.append(hist.history['loss'])
                loss_valid.append(hist.history['val_loss'])
            loss_train = flatten_list(loss_train)
            loss_valid = flatten_list(loss_valid)
        
        loss_predict = predict_history.pred_hist
        if params['save']:
            self.model.save(params['save_name'])
            
        fit_out = {'loss_train':loss_train,
                    'loss_valid':loss_valid,
                    'loss_predict':loss_predict,
                    'num_params':self.model.count_params()}                    
        return fit_out
    
    def predict(self,X_test):
        #self.model.reset_states()
        y_pred = self.model.predict(X_test,batch_size=self.batch_size, verbose=self.verbose) #Make predictions
        return y_pred

class kalman_decoder:
    """
    Kalman filter decoding algorithm
    """
    def __init__(self,regular=None,alpha_reg=0):
        self.regular = regular # type of regularisation
        self.alpha_reg = alpha_reg # regularisation constant

    def fit(self,X_train,Y_train,**params):
        self.regular = params['regular']
        self.alpha_reg = params['alpha_reg']
        if self.regular=='l1':
            regres = Lasso(alpha=self.alpha_reg)            
        elif self.regular=='l2':
            regres = Ridge(alpha=self.alpha_reg)
        elif self.regular=='l12':
            regres = ElasticNet(alpha=self.alpha_reg)
        else:
            regres = LinearRegression()
        
        X = Y_train 
        Z = X_train 
        nt = X.shape[0]              
        X1 = X[:nt-1,:] 
        X2 = X[1:,:] 
        
        regres.fit(X1,X2)
        A = regres.coef_ # shape (2, 2)
        W = np.cov((X2-np.dot(X1,A.T)).T) # np.cov input (features,samples), output shape (2, 2)
        regres.fit(X,Z)
        H = regres.coef_ # shape (96, 2)
        Q = np.cov((Z-np.dot(X,H.T)).T) # shape (96,96)
        params = [A,W,H,Q] # ----> should be in matrix form (not numpy)
        self.model = params
    
    def predict(self,X_kf_test,y_test):
        # extract parameters
        A,W,H,Q = self.model

        X = np.matrix(y_test.T)
        Z = np.matrix(X_kf_test.T)

        # initialise states and covariance matrix
        num_states = X.shape[0] # dimensionality of the state
        states = np.empty(X.shape) # keep track of states over time (states is what will be returned as y_pred)
        P_m = np.matrix(np.zeros([num_states,num_states]))
        P = np.matrix(np.zeros([num_states,num_states]))
        state = X[:,0] # initial state
        states[:,0] = np.copy(np.squeeze(state))

        # get predicted state for every time bin
        for t in range(X.shape[1]-1):
            # do first part of state update - based on transition matrix
            P_m = A*P*A.T+W
            state_m = A*state

            # do second part of state update - based on measurement matrix
            try:
                K = P_m*H.T*inv(H*P_m*H.T+Q) #Calculate Kalman gain
            except np.linalg.LinAlgError:
                K = P_m*H.T*pinv(H*P_m*H.T+Q) #Calculate Kalman gain
            P = (np.matrix(np.eye(num_states))-K*H)*P_m
            state = state_m+K*(Z[:,t+1]-H*state_m)
            states[:,t+1] = np.squeeze(state) #Record state at the timestep
        y_pred = states.T
        return y_pred

class wiener_decoder:
    """
    Wiener filter decoding algorithm
    """
    def __init__(self,regular=None,alpha=0):
        self.regular = None # type of regularisation
        self.alpha = 0 # regularisation constant

    def fit(self,X_train,y_train,**params):
        self.regular = params['regular']
        self.alpha = params['alpha']
        if self.regular=='l1':
            self.model = Lasso(alpha=self.alpha)            
        elif self.regular=='l2':
            self.model = Ridge(alpha=self.alpha)
        elif self.regular=='l12':
            self.model = ElasticNet(alpha=self.alpha)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train, y_train)

    def predict(self,X_test):
        y_pred = self.model.predict(X_test) #Make predictions
        return y_pred
