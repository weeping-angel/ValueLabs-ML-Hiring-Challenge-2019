import re
import pandas as pd
import random

df = pd.read_csv('Train.csv')

l=[]

for i in range(0, len(df)):
    answer = df.iloc[i]['answer_text']
    dists = df.iloc[i]['distractor']
    
    Di = re.split(r'[\'\"],', dists)

    Di = [d.strip(' ') for d in Di]
    Di = [d.strip('"') for d in Di]
    Di = [d.strip("'") for d in Di]
    Di = set([d.strip(' ') for d in Di])
    
    for d in Di:
        l.append([answer, d])


l = random.sample(l, len(l))   
fw = pd.DataFrame(l, columns=['answer_text', 'distractor'])
fw.to_csv('Train_new.csv', index=None, sep='\t')
    
    
    
    
    
    
    
    
