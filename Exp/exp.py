import nltk 
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd

df = pd.read_csv('Results.csv')
tokenizer = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')

def dict_has(word):
    if not wordnet.synsets(word):
        return False
    else:
        return True

def get_synonyms(word):
    synonyms = [] 
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonyms.append(l.name()) 
            
    return list(set(synonyms))
    
def remove_stop_words(text):
    l = [word for word in text.split(' ') if word not in stop_words]
    new_text = ' '.join(l)
    return new_text
    
def preprocess(text):
    text = text.lower()
    words = tokenizer.tokenize(text)
    new_text = ' '.join(words)
    return new_text
    
def make_syn_sent(sentence, remove_stop_words=False):
    sentence = preprocess(sentence)
    if remove_stop_words:
        sentence = remove_stop_words(sentence)
    valid_words = [word for word in sentence.split(' ') if dict_has(word)]
    new_words = [get_synonyms(word)[0] for word in valid_words]
    new_sentence = "'" + ' '.join(new_words) + "'"
    return new_sentence
   
def main():
    df = pd.read_csv('Results.csv')
    dist = []
    for i in df['answer_text']:
        dist.append(make_syn_sent(i))
        
    df['distractor']=dist
    df.to_csv('exp.csv', index=None)
    print('Head : \n', df.head())
  
 

