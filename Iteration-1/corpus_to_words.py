import io
from collections import Counter

with io.open('corpus', encoding='utf-8') as f:
    text = f.read().lower().replace('\n', ' \n ')
print('Corpus length in characters:', len(text))

text_in_words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
print('Corpus length in words:', len(text_in_words))

unique_words = set(text_in_words)
print('Unique Words : ', len(unique_words))

cnt=Counter(text_in_words)
#print(cnt)
tmp = cnt.copy()
for key in tmp.keys():
    if cnt[key] < 2:
        cnt.pop(key)

print('After Ignoring unfrequent words : ', len(cnt))


