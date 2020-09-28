# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import unicodedata
import re
import matplotlib.pyplot as plt
import os
import pandas as pd
import imageio
import requests
from zipfile import ZipFile
from model import Encoder, Decoder, LuongAttention, loss_func
from data import unicode_to_ascii, normalize_string


# Mode can be either 'train' or 'infer'
# Set to 'infer' will skip the training
MODE = 'infer'
#URL = 'http://www.manythings.org/anki/fra-eng.zip'
#FILENAME = 'fra-eng.zip'
BATCH_SIZE = 64
EMBEDDING_SIZE = 256
RNN_SIZE = 512
NUM_EPOCHS = 15

r=pd.read_csv('Results.csv')
test_sents = r['answer_text']

# Set the score function to compute alignment vectors
# Can choose between 'dot', 'general' or 'concat'
ATTENTION_FUNC = 'concat'
answers = []

lines = open('fra.tsv', 'r', encoding='utf-8').read()
#lines = lines.decode('utf-8')

raw_data = []
for line in lines.split('\n'):
    raw_data.append(line.split('\t'))
    
#raw_data = raw_data[:10000]

print(raw_data[-5:])
# The last element is empty, so omit it
raw_data = raw_data[:-1]

raw_data_en, raw_data_fr = list(zip(*raw_data))
raw_data_en = [normalize_string(data) for data in raw_data_en]

test_sent = [normalize_string(data) for data in test_sents]

raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en)

#en_tokenizer.fit_on_texts(test_sents)

data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,
                                                        padding='post')
print('Input sequences')
print(data_en[:2])

fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)
data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in,
                                                           padding='post')
print('Target input sequences')
print(data_fr_in[:2])

data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,
                                                            padding='post')
print('Target output sequences')
print(data_fr_out[:2])

dataset = tf.data.Dataset.from_tensor_slices(
    (data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(len(raw_data_en)).batch(
    BATCH_SIZE, drop_remainder=True)

en_vocab_size = len(en_tokenizer.word_index) + 1

encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, RNN_SIZE)


fr_vocab_size = len(fr_tokenizer.word_index) + 1
decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, RNN_SIZE, ATTENTION_FUNC)

# These lines can be used for debugging purpose
# Or can be seen as a way to build the models
initial_state = encoder.init_states(1)
encoder_outputs = encoder(tf.constant([[1]]), initial_state)
decoder_outputs = decoder(tf.constant(
    [[1]]), encoder_outputs[1:], encoder_outputs[0])
    
    
optimizer = tf.keras.optimizers.Adam(clipnorm=5.0)

@tf.function
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    loss = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_state_h, de_state_c = en_states

        # We need to create a loop to iterate through the target sequences
        for i in range(target_seq_out.shape[1]):
            # Input to the decoder must have shape of (batch_size, length)
            # so we need to expand one dimension
            decoder_in = tf.expand_dims(target_seq_in[:, i], 1)
            logit, de_state_h, de_state_c, _ = decoder(
                decoder_in, (de_state_h, de_state_c), en_outputs[0])

            # The loss is now accumulated through the whole batch
            loss += loss_func(target_seq_out[:, i], logit)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss / target_seq_out.shape[1]
    
def predict(test_source_text=None):
    if test_source_text is None:
        test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

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

    answer = ' '.join(out_words)
    answers.append(answer)
    print(answer)
    return np.array(alignments), test_source_text.split(' '), out_words


if not os.path.exists('checkpoints_luong/encoder'):
    os.makedirs('checkpoints_luong/encoder')
if not os.path.exists('checkpoints_luong/decoder'):
    os.makedirs('checkpoints_luong/decoder')

# Uncomment these lines for inference mode
encoder_checkpoint = tf.train.latest_checkpoint('checkpoints_luong/encoder')
decoder_checkpoint = tf.train.latest_checkpoint('checkpoints_luong/decoder')

if encoder_checkpoint is not None and decoder_checkpoint is not None:
    encoder.load_weights(encoder_checkpoint)
    decoder.load_weights(decoder_checkpoint)

if MODE == 'train':
    for e in range(NUM_EPOCHS):
        en_initial_states = encoder.init_states(BATCH_SIZE)
        encoder.save_weights(
            'checkpoints_luong/encoder/encoder_{}.h5'.format(e + 1))
        decoder.save_weights(
            'checkpoints_luong/decoder/decoder_{}.h5'.format(e + 1))
        for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
            loss = train_step(source_seq, target_seq_in,
                              target_seq_out, en_initial_states)

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    e + 1, batch, loss.numpy()))

        try:
            predict()

            predict("How are you today ?")
        except Exception:
            continue


if not os.path.exists('heatmap'):
    os.makedirs('heatmap')

filenames = []

for i, test_sent in enumerate(test_sents):
    test_sequence = normalize_string(test_sent)
    alignments, source, prediction = predict(test_sequence)
    attention = np.squeeze(alignments, (1, 2))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='jet')
    ax.set_xticklabels([''] + source, rotation=90)
    ax.set_yticklabels([''] + prediction)

    filenames.append('heatmap/test_{}.png'.format(i))
    plt.savefig('heatmap/test_{}.png'.format(i))
    plt.close()

with imageio.get_writer('translation_heatmaps.gif', mode='I', duration=2) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

r['distractor'] = answers
r.to_csv('Predictons_enc_dec.csv', index=None)
