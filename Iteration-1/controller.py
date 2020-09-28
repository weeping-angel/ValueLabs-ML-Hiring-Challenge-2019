# Main File that will run the whole Module
from keras.models import load_model
from data import Data
from model import build_lstm_model
from os.path import exists
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import scipy

class Controller:
    def __init__(self):
        self.data = Data()
        self.vocab_size, self.embedding_size, self.pretrained_weights = self.data.load_embedding()
        self.train_x, self.train_y, self.test_x, self.test_y = self.data.split()
        self.model_file = 'model.hdf5'
        print('Vocab Size = ', self.vocab_size)

    def train(self, batch_size=64, epochs=10):
        print('Starting training ...')
        x = self.train_x # Input
        y = self.train_y # Output
        if exists(self.model_file):
            #load model file
            model = load_model(self.model_file)
        else:
            model = build_lstm_model(self.vocab_size, self.embedding_size, self.pretrained_weights)

        model.fit([x],y, batch_size=batch_size, epochs=epochs, verbose=1)
        print('Model Training Completed')

        model.save(self.model_file)
        print('Trained Model saved as : ', self.model_file)

    def predict(self):
        print('Starting prediction ...')
        if exists(self.model_file):
            x = self.data.get_train_data()[-90:, 0]
            y = self.data.get_train_data()[-90:, 1]
            preds = []
            model = load_model(self.model_file)
            for question in x:
                answer = self.data.generate_next(model, question)
                preds.append(answer)
            
            print('Some predictions : ', preds[:3], '\n\n')
            return y, preds
        else:
            print("Model File doesn't exist!!!")
            exit()


    def predict_on_test_file(self, temperature=0.3):
        print('Starting prediction on test file...')
        if exists(self.model_file):
            test_data = self.data.get_test_data()
            x = test_data['question']
            preds = []
            model = load_model(self.model_file)
            i=0
            for question in x:
                answer = self.data.generate_next(model, question, num_generated = 6, temperature=temperature)
                preds.append(answer)
                i=i+1
                if i%100==0: 
                    print(i,' Predictions Completed')

            print('Total_predictions : ', len(preds))

            if not exists('Results.csv'):
                tmp = open('Results.csv', 'w')
                tmp.close()

            result_file = pd.read_csv('Results.csv')
            result_file['distractor'] = preds
            print('Prediction CSV Head : ', result_file.head())
            result_file.to_csv('Predictions.csv', index=None)
            print('Prediction File is ready.')
        else:
            print("Model File doesn't exist!!!")

    def compare(self, exact_values, predictions):
        text = open('corpus', 'r', encoding='utf-8')
        vectorizer = CountVectorizer()
        vectorizer.fit(text)
        count_exact_values = [vectorizer.transform(sent.split(' ')) for sent in exact_values] # ndarray or sparse array, shape: (n_samples_X, n_features)
        count_predictions = [vectorizer.transform(sent.split(' ')) for sent in predictions] # ndarray or sparse array, shape: (n_samples_Y, n_features)
        cosine_similarity = [(1-scipy.spatial.distance.cosine(i,j)) for i,j in zip(count_exact_values, count_predictions)]
        score = 100 * sum(cosine_similarity)/len(cosine_similarity)
        #cosine_similarity_scores = [cosine_similarity(i,j) for i,j in zip(count_exact_values, count_predictions)]
        #score = [100 * sum(i)/len(i) for i in cosine_similarity_scores]
        return sum(score)

    def run(self):
        self.train()
        self.predict_on_test_file()
        #self.train()
        #exact_values, predictions = self.predict()
        #score = self.compare(exact_values, predictions)
        #print(score)
