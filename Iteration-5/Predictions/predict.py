import pandas as pd
import tensorflow as tf
import numpy as np
import os

from data import unicode_to_ascii, normalize_string
from model import Encoder, Decoder, LuongAttention

BATCH_SIZE = 64
EMBEDDING_SIZE = 256
RNN_SIZE = 512
NUM_EPOCHS = 15
ATTENTION_FUNC = 'concat'

r=pd.read_csv('Results.csv')
test_sents = r['answer_text'] 
dists = []

lines = open('Train.tsv', 'r', encoding='utf-8').read()

raw_data = []
for line in lines.split('\n'):
    raw_data.append(line.split('\t'))

print(raw_data[-5:])

raw_data = raw_data[:-1]


raw_data_en, raw_data_fr = list(zip(*raw_data))
raw_data_en = [normalize_string(data) for data in raw_data_en]
raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]

test_sent = [normalize_string(data) for data in test_sents] #New

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en)
data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,padding='post')

en_tokenizer.fit_on_texts(test_sents) #New

print('Input sequences')
print(data_en[:2])

fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)
data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in, padding='post')

print('Target input sequences')
print(data_fr_in[:2])

data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out, padding='post')

print('Target output sequences')
print(data_fr_out[:2])

dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(len(raw_data_en)).batch(BATCH_SIZE, drop_remainder=True)

en_vocab_size = len(en_tokenizer.word_index) + 1

encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, RNN_SIZE)

fr_vocab_size = len(fr_tokenizer.word_index) + 1
decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, RNN_SIZE, ATTENTION_FUNC)

initial_state = encoder.init_states(1)
encoder_outputs = encoder(tf.constant([[1]]), initial_state)
decoder_outputs = decoder(tf.constant(
    [[1]]), encoder_outputs[1:], encoder_outputs[0])

def predict(test_source_text=None):
    if test_source_text is None:
        test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    
    if len(test_source_seq[0])==0:
        dists.append(test_source_text)
        return 0, test_source_text, test_source_text

    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

    de_input = tf.constant([[fr_tokenizer.word_index['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []
    alignments = []

    while True:
        de_output, de_state_h, de_state_c, alignment = decoder(
            de_input, (de_state_h, de_state_c), en_outputs[0])
        de_input = tf.expand_dims(tf.argmax(de_output, -1), 0)
        out_words.append(fr_tokenizer.index_word[de_input.numpy()[0][0]])

        alignments.append(alignment.numpy())

        if out_words[-1] == '<end>' or len(out_words) >= 20:
            break

    dist = ' '.join(out_words)
    dists.append(dist)
    return np.array(alignments), test_source_text.split(' '), out_words


encoder.load_weights('checkpoints_luong/encoder/encoder.h5')
decoder.load_weights('checkpoints_luong/decoder/decoder.h5')

for i, test_sent in enumerate(test_sents):
    test_sequence = normalize_string(test_sent)
    alignments, source, prediction = predict(test_sequence)
    if i%100==0: print(i, ' Predictions completed!')
    
    
dists = ["'"+d[:-6]+"'" for d in dists]
r['distractor']=dists
r.to_csv('Predictions.csv', index=None)
