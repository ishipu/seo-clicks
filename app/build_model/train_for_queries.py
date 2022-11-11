from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Activation, Dropout


def train_model_for_queries(trainX, trainY):
    print('\nModel preparation...\n')
    model = Sequential() 
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1],trainX.shape[2]),return_sequences= False)) #,return_sequences= True))
    model.add(Dropout(0.25))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    print('\nModel fiting...\n')
    model.fit(trainX, trainY, epochs=50, batch_size=8, verbose=0)

    return model