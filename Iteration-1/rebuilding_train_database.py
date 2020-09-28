# Script for restructuring database

import pandas as pd
import numpy as np

def preprocess(line):
	punctuations = '''!()[]{};':"\,<>./?@#-$=`+%^&*_~'''
	no_punc_line=""
	
	line = line.lower() #lowercasing
	
	for letter in line: #Removing Punctuations
		if letter not in punctuations and letter!='\n':
			no_punc_line = no_punc_line + letter
		else:
		    no_punc_line = no_punc_line + ' '
		
	no_punc_line = no_punc_line.strip()
	no_punc_line = no_punc_line.strip('\'')
	
	return no_punc_line.strip()
	
def seperate(dist):
    #ret = [d.strip('') for d in dist.split('\'') if len(d)>2]
    return dist.split(', \'')

df = pd.read_csv('Train.csv')
new_db = open('Train_new.csv', 'w')

for i in range(0, len(df)):
    question = preprocess(df['question'][i])
    new_db.write('"'+ question +'",')
    new_db.write('"'+preprocess(df['answer_text'][i])+'"\n')
    for d in seperate(df['distractor'][i]):
        f=preprocess(d)
        if len(f)!=0:
            new_db.write('"'+question+'",')
            new_db.write('"' + f + '"\n')

new_db.close()

# Shuffling new database
df = pd.read_csv('Train_new.csv')
print('Shape of new structure of Database : ', df.shape)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('Train_new_shuffled.csv', index=False)
    
