import nltk
from collections import Counter
from gensim.models import KeyedVectors
import numpy as np
import os
from scipy import spatial

class FeatureVector:
    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        
    def make_vector(self, ques, ans, dist):
        ques = str(ques)
        ans = str(ans)
        dist = str(dist)
        emb_sim_ad = self.emb_sim(ans, dist)
        freq_d = self.freq(dist)
        freq_a = self.freq(ans)
        #wiki_sim = self.wiki_sim(ans, dist)
        emb_sim_qd = self.emb_sim(ques, dist)
        abs_suffix_len = self.abs_suffix_len(ans, dist)
        rel_suffix_len_d = self.rel_suffix_len(dist, abs_suffix_len)
        rel_suffix_len_a = self.rel_suffix_len(ans, abs_suffix_len)
        token_sim = self.token_sim(ans, dist)
        ed = self.ed(ans, dist)
        
        vec = [emb_sim_ad, freq_d, freq_a, emb_sim_qd, abs_suffix_len, rel_suffix_len_d, rel_suffix_len_a, token_sim, ed]
        #vec = np.array(vec)
        return vec
        
    def pos_sim(self,a,b):
        pos_tags_a = a
        pos_tags_b = b
        ret = nltk.jaccard_distance(pos_tags_a,pos_tags_b)
        return ret
        
    def emb_sim(self,a,b):
        a=str(a)
        b=str(b)
        
        feat_vec_a = np.zeros((300, ), dtype=np.float32)
        feat_vec_b = np.zeros((300, ), dtype=np.float32)
        
        a_words = 0
        b_words = 0
        
        for word in a.split(' '):
            if word in self.model.vocab:
                feat_vec_a = np.add(feat_vec_a, self.model[word])
                a_words = a_words + 1
            
        for word in b.split(' '):
            if word in self.model.vocab:
                feat_vec_b = np.add(feat_vec_b, self.model[word])
                b_words = b_words + 1
            
        if a_words!=0 : avg_vec_a = np.divide(feat_vec_a, a_words)
        else: return 0
        if b_words!=0 : avg_vec_b = np.divide(feat_vec_b, b_words)
        else: return 0
        
        try:
            sim = 1 - spatial.distance.cosine(avg_vec_a, avg_vec_b)
        except ValueError:
            sim = 0
        
        return sim
        
    def ed(self,a,b):
        ret = nltk.edit_distance(a,b)
        return ret
        
    def token_sim(self,a,b):
        a = set(a.split(' '))
        b = set(b.split(' '))
        ret = nltk.jaccard_distance(a,b)
        return ret
        
    def freq(self,a):
        a = Counter(a.split())
        a = a.values()
        ret = sum(a)/len(a)
        return ret
        
    def abs_suffix_len(self,a,b):
        list_of_strings = [a,b]
        reversed_strings = [' '.join(s.split()[::-1]) for s in list_of_strings]
        reversed_lcs = os.path.commonprefix(reversed_strings)
        lcs = ' '.join(reversed_lcs.split()[::-1])
        return len(lcs)
        
    def rel_suffix_len(self,a, suffix_len):
        ret = suffix_len/len(a)
        return ret
