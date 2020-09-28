import gensim.models as g
from sklearn.manifold import TSNE
import re
import pandas as pd
import matplotlib.pyplot as plt

modelPath="word2vec.model"
model = g.Word2Vec.load(modelPath)

vocab = list(model.wv.vocab)
X = model[vocab]

print(len(X))
print(X[0])
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)
