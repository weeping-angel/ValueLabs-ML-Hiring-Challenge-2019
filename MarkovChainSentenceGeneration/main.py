import random 
import pandas as pd
import os

df = pd.read_csv('Results.csv')
model = {}
answers=[]
step=5
h=0
for i in range(0,len(df), step):
    data = df['answer_text'][i:i+step]
    for line in data:
        line = line.lower().split()
        for i, word in enumerate(line):
            if i == len(line)-1:   
                model['END'] = model.get('END', []) + [word]
            else:    
                if i == 0:
                    model['START'] = model.get('START', []) + [word]
                model[word] = model.get(word, []) + [line[i+1]] 

    answer = ''
    for i in range(step*3):
        generated = []
        while True:
            if not generated:
                words = model['START']
            elif generated[-1] in model['END']:
                break
            else:
                words = model[generated[-1]]
            generated.append(random.choice(words))
            
        answer = answer + "'" + ' '.join(generated) + "',"
        if i%3==0: 
            answers.append(answer.strip(','))
            answer = ''

print('Total answers : ', len(answers))
print('Total questions : ', len(df))
df['distractor'] = answers
df.to_csv('exp.csv', index=None)
    
