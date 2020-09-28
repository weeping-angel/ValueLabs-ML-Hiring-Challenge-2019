import pickle

model = pickle.load(open('embedding-matrix.pickle', 'rb'))

def similar(word_vec, number):

    dst = (np.dot(model, word_vec)
           / np.linalg.norm(self.word_vectors, axis=1)
           / np.linalg.norm(word_vec))
           
    word_ids = np.argsort(-dst)

    return [(self.inverse_dictionary[x], dst[x]) for x in word_ids[:number]
            if x in self.inverse_dictionary]

l = similar('apple',1)
print(l)
