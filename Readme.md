# My 13 day journey on the streets of NLP

## Mission

Snippet from the Challenge Page : 

_MCQs are a widely-used question format that is used for general assessment on domain knowledge of candidates. Most of the MCQs are created as paragraph-based questions._

_A paragraph or code snippet forms the base of such questions. These questions are created based on the three or four options from which one option is the correct answer. The other remaining options are called distractors which means that these options are nearest to the correct answer but are not correct._

_You are provided with a training dataset of questions, answers, and distractors to build and train an NLP model. The test dataset contains questions and answers. You are required to use your NLP model to create up to three  distractors for each question-answer combination that is provided in the test data._

Genre: NLP

Problem type: Contextual Semantic Similarity, Auto generate Text-based answers 

## Approaches

After grasping the uniqueness of the given problem, I tried several different approaches in the quest of highest quality distractors. My definition for the perfect distractor was,

 Distractor : _A sentence in continuation of the question/phrase/stem which is highly similar, but not identical, to the answer/key._

My Top Five Approaches were:

1. **Iteration-1**: Using 2 layer RNN-LSTM network with word2vec embedding trained on the given dataset (NOT pretrained) using fixed length token vectors before making sequences(of fixed length) out of them, to feed into the neural network. The only flaw in this approach was, word2vec embedding vectors were not able to trained on the given dataset, due to its semi-structure nature. Principle Component Analysis (PCA) and 2D Scattergraph plotting showed the cluster formation of unrelated/all words in the corpus. Hence, moved to Iteration-2. (Accuracy: between 12% to 17% on evaluation criteria)

2. **Iteration-2**: Using 2 layer RNN-LSTM network with GloVe pre-trained embedding vectors. Using variable length token vectors to make smaller fixed length sequences and feeding it to the network. This approach was way better, but hit a roadblock when got stuck into a local minima and was slower in getting the context. Hence, moved to Iteration-3. (Accuracy: between 16% to 20% on evaluation criteria)

3. **Iteration-3**: Using 2 layer Bidirectional-LSTM layer with Glove pre-trained embedding vectors. Using variable length token vectors to make smaller fixed length sequences and feeding it to the network. This approach performed the best, but for predictions, optimum _temperature_ value was difficult to find. (Accuracy: above 17% on evaluation criteria)

4. **Iteration-4**: Using LogisticRegression and Random Forest Classifier (LR-RF) cascading model for ranking the highest quality distractors from the distractor pool. Distractor pool was made from the training dataset. This approach was inspired from the research paper _"Distractor Generation for Multiple Choice Questions Using Learning to Rank"_ by Chen Liang, Xiao Yang, Neisarg Dave, Drew Wham, Bart Pursel and C. Lee Giles. (Accuracy: between 15% to 18% on evaluation criteria) 

5. **Iteration-5**: Using Encoder-Decoder model with Luong Attention, previously used in the Neural Machine Translation (NMT), for generating distractor from the input of answer keys alone. This approach is inspired from the paper _"Attention is all you need"_ by the team of Google Brains et al.

Other than these, several informal trials were made on the basis of the insights gathered from the dataset. See **_Notes_** Section for more information.

Uploaded results were the combination of iterations 3, 4 and 5, which scored the highest accuracy.

### Data Restructuring and building corpus. 

It was quintessential to restructure the given training and testing data according to the formulation of the problem. For each approach, the following tiny scripts were written to convert the given _Test.csv_ and _Train.csv_ files to _Test_new.csv_ or _Train_new.csv_ files respectively.

`restructuring_training_data.py`

`restructuring_testing_data.py`

For combining both newly generated files, use the following script:

`Combining_train_test.py`

And for building the corpus of words common is testing and training dataset, following script can do the job.

`building_corpus.py`

### Data Preprocessing

As the saying goes, "Garbage In, Garbage Out". It could not hold more true than it does in case of Deep Learning and Machine Learning tasks like this. For the NLP domain, **NLTK** (Natural Language ToolKit) is my choice of the weapon.

I used it to remove unwanted symbols, punctuation and stopwords for the analysis and training purposes. Also, Tokenization, POS tagging, calculating Edit Distance and Jaccard Similarity ( In Iteration-4 ) were done using this package's tools.

Next step in data preprocessing was making vectors of the given question, answer and distrators.

#### Word Embedding Dilemma

Traditionally, people tend to use one-hot vectors for their smaller datasets in the text generation task. But for the Middle to larger datasets, it is absolutely necessary to use embeddings of fixed dimension to save memory and train the network efficiently. 

Here, I used GloVe (Global Vectors) and Word2Vec, both of dimension size = 300, separately in Iteration-1 and Iteration-(2,3) for converting my token vectors which will be passing through the _EmbeddingLayer_ in neural architecture to embedded vectors. These vectors will also be capturing the semantics of the sentence and context, unlike one-hot vectors.

Only problem here was, properly calculating loss and updating gradients. This can be done by using **_sparse_categorical_crossentropy_** as the loss function instead of _categorical_crossentropy_ and to use **_sparse_categorical_accuracy_** as the metric function.

#### Pickling and saving embeddings.

Since it was pretty expensive to create the whole word embedding matrix every time the program runs, I had to make the corpus of words included in the training and testing dataset both, and develop an embedding matrix, once and for all. 

After the creation of _embedding_matrix_ file using _pickle_, I could check its local existence and load it into the memory in relatively and significantly shorter period of time. Hence, speeding up the program.

#### Generating sequences

Now just before feeding our Neural Network, We need our data to be in the form of sequences. 

Consider a function _f_. For producing the output _y_, it needs _n_ number of inputs. Let inputs be represented with _x_. So mathematically, we can write this as 

_y  =  f ( x1, x2, x3, . . . , xn )_

So here, [x1, x2, x3, ..., xn] is the input sequence of length n.
and y is the output sequence of length 1. And _f_ is the function determined by the neural net.

In my python implementation of sequences, it was achieved using Numpy arrays. Numpy is a package of powerful tools for mathematical operations. And Numpy Arrays are what that goes into the neural networks, directly.

Shape of the input sequence: (total_samples, sequence_length, embedding_dimension)

Shape of the output_sequence: (total_samples, embedding_dimension)

Here, sequence length I used was n = 5.

In short, sequence of 5 words predicting the 6th word.

### Neural Network Architecture (Brief Overview)

For Iteration 1 and Iteration 2 (Sequential Model):

`Embedding Layer >> LSTM >> Dropout >> LSTM >> Dropout >> Dense`

For Iteration 3 (Sequential Model):

`Embedding Layer >> Bi-LSTM >> Dropout >> Bi-LSTM >> Dropout >> Dense`

For Iteration 4: No Neural nets were used. (LR-RF Model was used)

For Iteration 5 (Not Sequential Model):

`Encoder(Embedding >> LSTM)`

`LuongAttention`

`Decoder(Embedding Layer >> LSTM >> LuongAttention)`

Neural Architecture for Iteration 5 is harder to describe here due to its non-sequential nature. Although, you can see it in the _model.py_ file, where Encoder, Decoder and LuongAttention are implemented as classes. Each of them contain the layers in _call( )_ function, defined within each class.

### Hyperparameters

#### For Iteration 1, 2 and 3

`loss = sparse_categorical_crossentropy`

`optimizer = Adam`

`Learning rate = 0.0001`

`activation = softmax`

`metric = sparse_categorical_accuracy`

#### For Iteration 4

Logistic Regression Model

`solver = lbfgs` (Limited memory Broyden–Fletcher–Goldfarb–Shanno)

`C=1.0` (Inverse Regularization Parameter)

Random Forest Model

`Trees = 500` (n-estimtors)

`Maximum depth = 30` (max-depth)

`Minimum weight value of node = 0.1` (min_weight_fraction_leaf)

#### For Iteration 5

`loss = sparse_categorical_crossentropy`

`optimizer = Adam`

`clipnorm = 5.0`

`batch size = 64`

`Attention function = concat`

#### Defusing Gradient Explosions

This was the hiccup I faced when I moved to LSTM and Bi-directional LSTM networks. For successfully preventing the gradient explosions I reduced the batch size to half, **128**, decreased the learning rate to **0.0001**, used clip normalization = **1.0** and did clipping = **0.5** while updating the gradient values through _Adam_ Optimizer.

## Highly Modular Program Structure

I followed a mid-way approach between Object-Oriented and Procedural. I tried to develop the whole software as in the form of APIs, so they can be reused again and again. It's development took a bit more time in the beginning, but was worth doing in the long run. Because of this approach, my workflow process was much much faster and I could try many more things and approaches to this problem, since I did not have to develop certain things from the ground again. 

**Note**: This design scheme is implemented in Iteration 1, 2 and 3 only. For Iteration 4 and 5, more compact design scheme was chosen because of the lack of time and need for rapid development.

### Details

My Software consisted of 4 Main Python files, plus some small scripts for restructuring and preprocessing data and building corpus. 

1. _main.py_ : This file is at the top-most in the hierarchy of execution. It imports the controller file and uses it as required.

2. _controller.py_ : This file contains the higher level abstraction of the workflow processes, like training, testing, comparing and data_loading tasks. This is all within a class named _Controller_.

3. _data.py_ : This file contained the tasks related to data manipulation and file creations and deletions, such as data preprocessing, embedding matrix creation, sampling for making distinct predictions, and many other helper functions. All this is encapsulated in the form of a Class name _Data_.

4. _model.py_ : This file is where all the magic happens. This file contained the neural network architecture implementation in keras and all the hyperparameters in the form a single function.

Among all the above mentioned files, _controller.py_ is the one that is integrating all the components. So, it is recommended to use this file as API interface for executing the tasks as a whole. Although, all files are designed in this way, that they can be used independent of each other, very easily.

## Usage as Whole

Note: Applicable for Iteration 1, 2 and 3 only. For Iteration 4 and 5, more compact design was chosen and hence, usage of them is slightly different.

#### Files in the package. (Iteration -1,2,3)

1. data.py
2. controller.py
3. model.py
4. main.py
5. requirements.txt
6. Train.csv
7. restructuring_training_data.py
8. Test.csv
9. Results.csv
10. building_corpus.py
11. restucturing_testing_data.py
12. Combining_train_test.py
13. run.sh

#### Files in the package. (Iteration-4)

1. feature_vector.py
2. train.py
3. test.py
4. test_multithreading.py
5. to_feature_vectors.py
6. restructure_training_data.py
7. formatting_predictions.py
8. make_distractor_pool.py
9. Train.csv
10. Results.csv
11. requirements.txt
12. run.sh

#### Files in the package. (Iteration-5)

1. data.py
2. model.py
3. train.py
4. predict.py
5. Train.tsv
6. Results.csv
7. requirements.txt
8. run.sh

### Prerequisites

Since the whole program is developed in python, Python3 is the top-most requirement.

All the essential packages can be installed by the following command.

`# pip install -r requirements.txt`

It is suggested to use a GPU-enabled device because training and prediction can take too long on CPU-only devices. For Utilizing GPU, use CUDA software provided by nVidia for nVidia Graphics.

### Running the program

For running the whole program at once, including installing prerequisites, setting up data and environment, training and testing. Run the following command

`# sh run.sh`

For manually running the program only, enter the following command in shell.

`# python3 main.py`

## Usage as API

For Iteration 1, 2 and 3 only.

### How to interface

`>>> from controller import Controller`

`>>> cnt=Controller()`

### How to train your network

`>>> cnt.train(batch_size=128, epochs=10)`

### How to make predictions

`>>> cnt.predict_on_test_file()`

## Notes

In my point of view, no offense, Evaluation criteria for this problem was not well suited to this particular task. Because Cosine Similarity of Count Vectors do not incorporate the semantic idea that the sentences are trying to convey. 

A Better approach would be to use GloVe Vectors or any other Word Embeddings that can truly capture the semantic meaning and then calculate Cosine Similarity.

### Loopholes

Along the way, I found some mathematical loopholes and exploited it to test my hypothesis. _(Source code is included in folder named "Informal Approaches")_

1. _32.13%_ accuracy on copying the same correct answer plus top 6 most frequent words - "the, of, to, a, is, in".

2. _28.49%_ accuracy on just copying the same correct answer as the distractor 3 times.

3. _20.01%_ accuracy by printing the top 8 most frequent words from the corpus, 3 times as distractor.

4. _16.63%_ accuracy by printing the most frequent word, 'the' 6 times in a string, 3 times as distractor.

These approaches depicts the fact that the above mentioned distractors will have lesser or no chance in distracting the person who is appearing for MCQ test and sees them as the options. But still they have higher accuracy according to the Evaluation Criteria. Whereas, options that have significantly higher chances in confusing the person have lower score on Evaluation Scale.

In short, Evaluation criteria loses its credibility at this point.

### One Informal Approach that didn't score well...

"Markov Chain Sentence generation" technique gave pretty good results too. But unfortunately, didn't got high enough score to be presented as a formal approach. It deserves some recognition and further development for the remarkable job it did. So it was worth mentioning. 

_Code for this approach is also included in the package._

### Development and Training Environment 

For Development, I used my own computer with the following configuration : 

1. 8 GB of RAM
2. 2 GB of nVidia GeForce Graphics
3. i5-6200U CPU @ 2.30GHz × 4
4. 1 TB of HDD

For the Training purpose, I naturally inclined towards Google's Colab (with GPU-enabled Notebook) Cloud computing environment because their interface is super flexible and easy to understand and does not hinder the workflow in any way. Its integration with Google drive gives it an added advantage too.

## Conclusion

On the evaluation criteria, my "informal" approaches did the best, whereas, in the real-world application of the problem, my formal approaches - Iteration-(3,4 and 5) - won the race jointly. As explained earlier, in the _Notes_ Section, that **evaluation criteria is biased towards the word frequency**, it cannot correctly calculate the contextual similarity and hence, whether my deep learning models in Iteration-(1,2,3 and 5) are scalable in terms of learning is undetermined. Although, after reading many of the generated distractor by myself, I conclude that the results are up-to-the-mark.

## Final Words of Acknowledgement

I immensely enjoyed the challenge and learnt a lot of new things along the way. I'll be looking for some more challenging tasks in the future. I want to thank ValueLabs and their team for their efforts. Hope they find the best candidate, no less than me.

## Bibliography

1. _"Distractor Generation for Multiple Choice Questions Using Learning to Rank"_ by Chen Liang, Xiao Yang, Neisarg Dave, Drew Wham, Bart Pursel and C. Lee Giles.

2. _"Random Forest for label ranking"_ by Yangming Zhou and Guoping Qiu.

3. _"Attention is all you need"_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.