import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from os.path import exists

class Data:
    def __init__(self):
        '''
        Load Data
        '''
        print('\n Loading Data ...')
        self.train_data = self.get_whole_data() #self.get_train_data()
        self.test_data = self.get_test_data()
        self.word_model = None
        print('Shape of Loaded Data : ', self.train_data.shape)
        #print('Data Head : ', self.train_data.head())

    def split(self):
        '''
        Splitting data into 9:1 ratio 
        for training and testing respectively.
        '''
        print('\n Splitting Data ...')
        max_question_len = 20
        max_answer_len = 15

        total_samples = len(self.train_data)
        training_samples = int(total_samples*1)

        x = np.zeros([total_samples, max_question_len], dtype=np.int32)
        y = np.zeros([total_samples, max_answer_len], dtype=np.int32)

        print('x.shape = ', x.shape)
        print('y.shape = ', y.shape)

        s=0 #default value = 0 
        try:
            for question, answer in np.array(self.train_data):
                for i, word in enumerate(question.split(' ')):
                    if len(word)>=1 and i<max_question_len: 
                        x[s,i] = self.word2idx(word)
                for j, word in enumerate(answer.split(' ')):
                    if len(word)>=1 and j<max_answer_len: 
                        y[s,j] = self.word2idx(word)
                s=s+1
        except KeyError:
            print('\n Some KeyError in the following : ')
            print('\t Questions ['+ str(s) +'] = ',question)
            print('\t Answer ['+ str(s) +'] = ',answer)
            
        print('x[-1] = ', x[-1])
        print('y[-1] = ', y[-1])

        seq_x = []
        seq_y = []
        for q,a in zip(x,y):
            seq = np.concatenate((q,a))
            for i in range(0,max_answer_len):
                seq_x.append(seq[i:i+max_question_len])
                seq_y.append(seq[i+max_question_len])

        train_x = np.array(seq_x[:training_samples])
        train_y = np.array(seq_y[:training_samples])
        test_x = np.array(seq_x[training_samples:])
        test_y = np.array(seq_y[training_samples:])

        return train_x, train_y, test_x, test_y


    def load_embedding(self):
        '''
        Doing data preprocessing
        '''
        w2v_file = 'word2vec.model'
        # Word2Vec Embedding
        if exists(w2v_file):
            print('\n Embeddings already exists!')
            self.word_model = Word2Vec.load(w2v_file)
        else:
            print('\n Creating Word2vec Embeddings')
            documents = [' '.join(list(d)) for d in np.array(self.train_data)]
            #documents = documents + [' '.join(list(d)) for d in np.array(self.test_data)]
            sentences = [[word for word in document.lower().split()] for document in documents]
            self.word_model = Word2Vec(sentences, size=200, min_count = 1, window = 5, iter=100)
            self.word_model.save(w2v_file)
            print('Word2Vec model saved! - ', w2v_file)
        
        pretrained_weights = self.word_model.wv.syn0
        vocab_size, embedding_size = pretrained_weights.shape

        return vocab_size, embedding_size, pretrained_weights

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
      return self.word_model.wv.vocab[word].index

    def idx2word(self, idx):
      return self.word_model.wv.index2word[idx]

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
        #print('Word Indexes : ', word_idxs)
        ans_idxs = []
        for _ in range(num_generated):
            prediction = model.predict(x=np.array(word_idxs))
            idx = self.sample(prediction[-1], temperature=temperature)
            ans_idxs.append(idx)
        return ' '.join(self.idx2word(idx) for idx in ans_idxs)