import pickle
import pandas as pd
import random
import numpy as np
from feature_vector import FeatureVector
from threading import Thread

fv = FeatureVector()

lr_model = pickle.load(open('lr_model', 'rb'))
rf_model = pickle.load(open('rf_model', 'rb'))

fpred=0 # Failed Prediction count
df = pd.read_csv('Results.csv')
    
D = open('distractor_pool', 'r', encoding='utf-8').readlines()
D = list(set([d.strip('\n') for d in D if d!='\n' and d!=' \n']))
#print(D[:10])

def thread_func(starting_point, ending_point):
    prediction_file = open('predictions_'+str(starting_point)+'-'+str(ending_point)+'.csv', 'w', encoding='utf-8')
    for i in range(starting_point,ending_point):
        
        q = df.iloc[i]['question']
        a = df.iloc[i]['answer_text']
        final_dists = []
        pool_size = 1000
        #print('Question : ', q)
        #print('Answer : ', a)
        
        while len(final_dists)==0:
            vectors = []
            final_dists=[]
            if pool_size > len(D):
                print('Pool size exceeded! writing generic answer.')
                prediction_file.write(str(i)+','+ ','.join([a.strip('\n'),a.strip('\n'),a.strip('\n')]) + '\n')
                final_dists = [a,a,a]
                fpred = fpred + 1
            else:
                dists = random.sample(D, pool_size)
                for d in dists:
                    print('Distractor : ', d)
                    vec = fv.make_vector(q,a,d)
                    vectors.append(vec)
                   
                vectors = np.array(vectors)
                
                try:
                    preds = lr_model.predict(vectors)
                except ValueError:
                    print("ValueError: \n")
                    print('Vector Shape : ', vectors.shape)
                    print('Vector : \n')
                    for vector in vectors:
                        if float('nan') in vector:
                            print(vector)
                    exit(-1)
                    
                idxs=[x for x in range(0, len(preds)) if preds[x]==1]
                
                if len(idxs)!=0 : #Checking output from first stage ranker
                    vectors_2 = vectors[idxs]
                
                    pred_prob = lr_model.predict_proba(vectors_2)
                    pred_prob = pred_prob[:, 1]
                    idxs = pred_prob.argsort()[-100:] if len(pred_prob.argsort())>=100 else pred_prob.argsort()
                    
                    vectors_2 = vectors[idxs]
                    
                    preds = rf_model.predict(vectors_2)
                    idxs_2 = [x for x in range(0, len(preds)) if preds[x]==1]
                    if len(idxs_2)!=0 : #Checking output from second stage ranker
                        vectors_2 = vectors_2[idxs_2]
                        pred_prob = rf_model.predict_proba(vectors_2)[:, 1]
                        idxs_2 = pred_prob.argsort()[-3:] if len(pred_prob.argsort())>=3 else pred_prob.argsort()
                        
                        idxs = idxs[idxs_2]
                        final_dists = np.array(list(dists))[idxs]
                    
                        #print(final_dists)

                        prediction_file.write(str(i)+','+ ','.join(['\''+d.strip('\n')+'\'' for d in final_dists]) + '\n')
                        
                        if i%10==0: # Showing Progress
                            print(i+1, ' Predictions completed')

                    else:
                        print('\t\tNo Viable Distractors found in current pool using RF, Retrying ...')
                        pool_size = pool_size * 2
                        print('\t\tNew pool size = ', pool_size)
                else:
                    print('\tNo Viable Distractors found in current pool using LR, Retrying ...')
                    pool_size= pool_size * 2
                    print('\tNew pool size = ', pool_size)
    
    prediction_file.close()
    print('Predictions file is ready!')
    
threads = []
num_threads = 2
samples = int(len(df)/num_threads) if len(df)%num_threads==0 else 1350
for i in range(0, len(df)-samples, samples):
    thread = Thread(target=thread_func, args=[i, i+samples])
    thread.start()
    threads.append(thread)
    
for thread in threads:
    thread.join()
    
print('Number of failed predictions = ', fpred)
