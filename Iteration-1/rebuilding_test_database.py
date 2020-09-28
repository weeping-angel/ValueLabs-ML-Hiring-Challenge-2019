# Script for restructuring database

import pandas as pd
import numpy as np

def preprocess(line):
	punctuations = '''!()[]{};':"\,<>./?@#-=`+$%^&*_~'''
	no_punc_line=""
	
	line = line.lower() #lowercasing
	
	for letter in line: #Removing Punctuations
		if letter not in punctuations and letter!='\n':
			no_punc_line = no_punc_line + letter
		else:
		    no_punc_line = no_punc_line + ' '
	
	return no_punc_line.strip()

df = pd.read_csv('Test.csv')
new_db = open('Test_new.csv', 'w')

new_db.write('question,answer_text\n')

for i in range(0, len(df)):
    question = preprocess(df['question'][i])
    answer = preprocess(df['answer_text'][i])
    new_db.write('"'+ question +'",')
    new_db.write('"'+ answer +'"\n')

new_db.close()
    
