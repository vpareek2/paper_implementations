# Implementation of the paper "Efficient Estimation of Word Representations in Vector Space"

# Veer Pareek - 07/30/2024

"""
Some notes while reading:

This paper introduces a technique for representing words as vectors of numbers. Designed to caputer the contextual meaning of words with more accuracy than previous apporaches. 

Model Architecture:
Distributed representations of words learned by neural networks -> performs significantly better than LSA for preserving linear regularities among words. LDA is computationally expensive on large datasets. Goal is to maximize accuracy while minimizing computational complexity.

O = E * T * Q, 

where E is n_epochs, T is n_words in the training set, and Q is defined differently for each model architecture. A common choice is E = 3-50 and T is up to one billion. All models are trained using stochastic gradient descent and backpropogation.

The work in the paper looks at NNLM and RNNLM. In this paper the authors propose 2 new model architectures.

New architectures find that a neural network language model can be successfully trained in 2 steps: first, continuous word vectors are learned using simple model, and then the N-gram NNLM is trained on top of these distributed representations of words. 

Model Architecture 1: Continuous Bag-of-Words Model
Similar to the feedforward NNLM, where the non-linear hidden layer is removed and the projection layer is shared for all words (not j projection matrix). Thus, all words get projected into the same position (average the vectors). It's called bag-of-words because the order of words doesn't influence projection. They also use words from the future. Best performance on the task by introducing a log-linear classifier with four futurew and four history words at the input, where the training criterion is to correctly classify the current (middle) word. Training complexity:

Q = N * D + D * log2(V)

This is CBOW, unlike standard bag-of-words it uses continuous distributed represntation of the context.

Model Architecture 2: Continuous Skip-gram Model
Similar to CBOW, instead of predicting the current word based on the context, it tries to maximize classification of a word based on another word in the same sentence. Uses each current word as an input to a log-linear classifier with continuous projection layer, and predict words within a certain range before and after the current word. Training complexity:

Q = C * (D + D * log2(V))

where C is the max distance of words. Thus, if we choose C = 5, for each training word, we will select randomly a number R in range <1;C> then use R words from history and R words from the future of the word as correct labels.

"""

# Preprocess the data
# Tokenize the text corpus into individual words
# Build a vocabulary, assigning a unique index to each word
# Create input/output paris for training based on the context window size (e.g., 2-word context

# Define the CBOW model
# Initialize an embedding layer that maps words to vectors
# For each input, average the word vectors within the context window
# Pass the averaged vector through a fully connected layer with output size equal to the vocabulary size

# Define the Skip-gram model:
# Initialize an embedding layer that maps words to vectors
# For each input word, pass its vector through a fully connected layer with output size equal to the vocabulary size
# Apply softmax activation to obtain probability distributions over the vocabulary for each training loop

# Define the training loop:
# Split data into batches
# For each batch, perform forward propagation (i.e. compute predictions) and calculate loss using cross-entropy loss function.
# Perform backpropagation to compute gradients with respect to model parameters
#Update the model parameteres using suitable optimization algorithm (e.g., stochastic gradient descent
