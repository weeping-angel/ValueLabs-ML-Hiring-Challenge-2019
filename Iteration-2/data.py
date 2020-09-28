import numpy as np
import pandas as pd
from os.path import exists
from keras.preprocessing.text import Tokenizer
import pickle

class Data:
    def __init__(self):
        '''
        Load Data
        '''
        print('\n Loading Data ...')
        self.train_data = self.get_whole_data() #self.get_train_data()
        self.test_data = self.get_test_data()
        self.tk = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
        self.vocab_size = 0
        print('\t Shape of Loaded Data : ', self.train_data.shape)
        #print('Data Head : ', self.train_data.head())

    def preprocess(self):
        print('\n Preprocessing Data ...')
        x = []
        y = []

        #Assigning Indices to the words
        try:
            for question, answer in np.array(self.train_data):
                x.append([self.word2idx(word) for word in question.split(' ') if len(word)>=1])
                y.append([self.word2idx(word) for word in answer.split(' ') if len(word)>=1])

        except KeyError as e:
            print('\n Some KeyError in the following : ')
            print('\t Questions = ',question)
            print('\t Answer = ',answer)
            print(e)

        print('\t Indices Assigned to Words!')

        # Creating Sequences
        seq_x = []
        seq_y = []
        seq_length = 5
        seq_step = 1
        for q,a in zip(x,y):
            seq = q + a
            for i in range(0, len(seq) - seq_length, seq_step):
                seq_x.append(seq[i:i+seq_length])
                seq_y.append([seq[i+seq_length]])

        # one_hot_y = np.zeros((len(seq_y), self.vocab_size+1), dtype=np.bool)
        # for i in range(0, len(seq_y)):
        #     one_hot_y[i, seq_y[i]] = 1


        # print('\t one-hot y[1] : ', one_hot_y[1])
        # print('\t {} Sequences created'.format(len(one_hot_y)))
        # NOTE : Memory Expensive operation ...... 
        # using sparse_categorical_crossentropy with sparse_categorical_accuracy metrics

        return np.array(seq_x), np.array(seq_y)


    def load_embedding(self, dimension=50):
        print('\n Loading Glove Embeddings and creating matrix ...')

        matrix_file = 'embedding_matrix.pickle'
        GLOVE_DIM = dimension

        data = open('corpus', 'r').readlines()
        self.tk.fit_on_texts(data)
        self.vocab_size = len(self.tk.word_index) + 1

        if not exists(matrix_file):

            glove_file = 'glove.6B.' + str(GLOVE_DIM) + 'd.txt'
            emb_dict = {}
            glove = open(glove_file, 'r', encoding='utf-8')
            for line in glove:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                emb_dict[word] = vector
            glove.close()

            print('\t Glove Embeddings Loaded!')

            emb_matrix = np.zeros((self.vocab_size, GLOVE_DIM))
            for w, i in self.tk.word_index.items():
                if i < self.vocab_size:
                    vect = emb_dict.get(w)
                    if vect is not None:
                        emb_matrix[i] = vect
                else:
                    break

            print('\t Embedding Matrix Created!')

            with open(matrix_file, 'wb') as f:
                pickle.dump(emb_matrix, f)

            print('\t Matrix Saved as : ', matrix_file)
        else:
            with open(matrix_file, 'rb') as f:
                emb_matrix = pickle.load(f)
            print('\t Embedding Matrix Loaded!')

        return self.vocab_size, GLOVE_DIM, emb_matrix

    def get_train_data(self):
        '''
        Returns data
        '''
        training_data = pd.read_csv('Train_new_shuffled.csv', index_col=None, header=None)
        return np.array(training_data)

    def get_test_data(self):
        '''
        Returns data
        '''
        testing_data = pd.read_csv('Test_new.csv')
        return testing_data

    def get_whole_data(self):
        db = pd.read_csv('DB.csv', index_col=None, header=None)
        return db

    def word2idx(self, word):
      return self.tk.word_index[word]

    def idx2word(self, idx):
      return self.tk.index_word[idx]

    def sample(self, preds, temperature=0.7):
        if temperature <= 0:
            return np.argmax(preds)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_next(self, model, text, num_generated=10, temperature=0.2):
        word_idxs = [self.word2idx(word) for word in text.lower().split()]
        #print('Question Word Indexes : ', word_idxs)
        ans_idxs = []
        for _ in range(num_generated):
            prediction = model.predict(x=np.array(word_idxs), batch_size=64)
            #print('Prediction : ', prediction)
            idx = self.sample(prediction[-1], temperature=temperature)
            #idx = prediction[-1]
            word_idxs.append(idx)
            ans_idxs.append(idx)
        
        #print('Answer Word Indexes : ', ans_idxs)
        answer = ' '.join([self.idx2word(idx) for idx in ans_idxs if idx!=0])
        return answer
