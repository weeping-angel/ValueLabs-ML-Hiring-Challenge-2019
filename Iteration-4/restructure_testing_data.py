import pickle
import pandas as pd
import numpy as np
from feature_vector import FeatureVector

fv = FeatureVector()

df = pd.read_csv('Results.csv')
D = open('distractor_pool', 'r', encoding='utf-8').readlines()
D = list(set([d.strip('\n') for d in D if d!='\n' and d!=' \n']))
print('Distractors in distractor_pool = ', len(D))
test_vec = open('Test_vec.csv', 'w', encoding='utf-8')

for i in range(0,len(df)):
    dists = D[:6800]
    q = df.iloc[i]['question']
    a = df.iloc[i]['answer_text']
    for d in dists:
        vec = fv.make_vector(q,a,d)
        test_vec.write(','.join(list(map(str,vec)))+'\n')
        
    print(i+1)
        
test_vec.close()
