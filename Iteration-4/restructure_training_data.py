import re
import pandas as pd
import random

df = pd.read_csv('Train.csv')
D = open('distractor_pool', 'r', encoding='utf-8').readlines()
D = set([d.strip('\n') for d in D])
#print('Length of D : ', len(D))
l=[]

for i in range(0, len(df)):
    question = df.iloc[i]['question']
    answer = df.iloc[i]['answer_text']
    dists = df.iloc[i]['distractor']
    
    Di = re.split(r'[\'\"],', dists)
    #print(Di)
    Di = [d.strip(' ') for d in Di]
    Di = [d.strip('"') for d in Di]
    Di = [d.strip("'") for d in Di]
    Di = set([d.strip(' ') for d in Di])
    
    #print('Length of Di : ', len(Di))
    #print(Di)
    
    neg_D = D - Di
    neg_D = random.sample(neg_D, len(Di))
    
    for d in Di:
        l.append([question, answer, d, 1])
        
    for d in neg_D:
        l.append([question, answer, d, 0])

        
fw = pd.DataFrame(l, columns=['question', 'answer_text', 'distractor', 'label'])
fw.to_csv('Train_new.csv', index=None)
    
    
    
    
    
    
    
    
