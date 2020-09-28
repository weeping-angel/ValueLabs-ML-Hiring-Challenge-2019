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

fresh_train = True

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

test_sent = [normalize_string(data) for data in test_sents]

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en)
data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,padding='post')

en_tokenizer.fit_on_texts(test_sents)

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
decoder_outputs = decoder(tf.constant([[1]]), encoder_outputs[1:], encoder_outputs[0])


def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss

optimizer = tf.keras.optimizers.Adam(clipnorm=5.0)

@tf.function
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    loss = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_state_h, de_state_c = en_states

        for i in range(target_seq_out.shape[1]):
            decoder_in = tf.expand_dims(target_seq_in[:, i], 1)
            logit, de_state_h, de_state_c, _ = decoder(
                decoder_in, (de_state_h, de_state_c), en_outputs[0])

            loss += loss_func(target_seq_out[:, i], logit)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss / target_seq_out.shape[1]


if not os.path.exists('checkpoints_luong/encoder'):
    os.makedirs('checkpoints_luong/encoder')
if not os.path.exists('checkpoints_luong/decoder'):
    os.makedirs('checkpoints_luong/decoder')


if fresh_train==False:
    encoder.load_weights('checkpoints_luong/encoder/encoder.h5')
    decoder.load_weights('checkpoints_luong/decoder/decoder.h5')


#Training
print('Starting Training')
for e in range(NUM_EPOCHS):
    en_initial_states = encoder.init_states(BATCH_SIZE)
    encoder.save_weights(
        'checkpoints_luong/encoder/encoder.h5')
    decoder.save_weights(
        'checkpoints_luong/decoder/decoder.h5')
    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in,
                          target_seq_out, en_initial_states)

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                e + 1, batch, loss.numpy()))
