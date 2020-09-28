from keras.layers import Dense, LSTM, Activation, Embedding, Dropout
from keras.models import Sequential

hparams = {
    'optimizer': 'adma',
    'loss': 'sparse_categorical_crossentropy',
    'activation': 'softmax'
}

def build_lstm_model(vocab_size, embedding_size, pretrained_weights):
    ''' 
    Neural Network Architecture
    '''

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights]))
    model.add(LSTM(units=embedding_size, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(units=vocab_size))
    model.add(Activation(hparams['activation']))
    model.compile(optimizer=hparams['optimizer'], loss=hparams['loss'])

    model.summary()
    return model


'''
model = Sequential()
model.add(Bidirectional(LSTM(128), input_shape=(SEQUENCE_LEN, len(words))))
if dropout > 0:
    model.add(Dropout(dropout))
model.add(Dense(len(words)))
model.add(Activation('softmax'))
'''