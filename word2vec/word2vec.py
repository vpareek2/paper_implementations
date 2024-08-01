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

import os
import requests
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch._dynamo as dynamo


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to download and extract the Text8 dataset
def download_text8(data_dir: str = './data', url: str = 'http://mattmahoney.net/dc/text8.zip') -> str:
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, 'text8.zip')
    text8_path = os.path.join(data_dir, 'text8')

    if not os.path.exists(text8_path):
        try:
            # Download the file
            print("Downloading Text8 dataset...")
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            with open(zip_path, 'wb') as f:
                f.write(response.content)
        
            # Extract the file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zip_path)
        except requests.RequestException as e:
            print(f"Error downloading the dataset: {e}")
            return None
        except zipfile.BadZipFile:
            print("Error: The downloaded file is not a valid zip file.")
            return None
    else:
        print("Text8 dataset already downloaded.")
    return text8_path
 
# Preprocess the data
def preprocess_text8(text8_path: str) -> Tuple[List[int], List[str]]:
    try:
        # Load the dataset
        with open(text8_path, 'r') as f:
            text = f.read()

        # Tokenize the text into individual words
        tokens = nltk.word_tokenize(text)
    
        # Build vocabulary and convert to indices
        vocab = build_vocabulary(tokens)
        token_indices = convert_tokens_to_indices(tokens, vocab)
        return token_indices, vocab
    except Exception as e:
        print(f"Error preprocessing the dataset: {e}")
        return None, None
    
# Tokenize the text corpus into individual words
def tokenize_text(text: str) -> List[str]:
	tokens = nltk.word_tokenize(text)
	return tokens

# Build a vocabulary, assigning a unique index to each word
def build_vocabulary(tokens: List[str]) -> List[str]:
	word_counts = Counter(tokens)
	vocab = list(word_counts.keys())
	return vocab

# Convert tokens to integer indicies using the vocab
def convert_tokens_to_indices(tokens: List[str], vocab: List[str]) -> List[int]:
    token_indices = [vocab.index(token) for token in tokens]
    return token_indices

# Create training data by generating pairs of input and output words
def create_training_data(token_indices: List[int], window_size: int = 2) -> List[Tuple[int, int]]:
    training_data = []
    for i, target in enumerate(token_indices):
        context = token_indices[max(0, i - window_size): i] + token_indices[i+1: min(len(token_indices), i + window_size + 1)]
        for context_word in context:
            training_data.append((target, context_word))
    return training_data

# Convert training data to pytorch tensors
def convert_training_data_to_tensors(training_data: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
	input_words = torch.tensor([pair[0] for pair in training_data])
	output_words = torch.tensor([pair[1] for pair in training_data])
	return input_words, output_words

# Define the CBOW model
class CBOWModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super(CBOWModel, self).__init__()
        # Initialize an embedding layer that maps words to vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    # For each input, average the word vectors within the context window
    # Pass the averaged vector through a fully connected layer with output size equal to the vocabulary size
    def forward(self, context_words: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(context_words)     # (batch_size, context_size, embedding_dim)
        avg_embeds = torch.mean(embeds, dim=1)      # (batch_size, embedding_dim)
        out = self.linear(avg_embeds)               # (batch_size, vocab_size)
        log_probs = F.log_softmax(out, dim=1)       # (batch_size, vocab_size)
        return log_probs

# Define the Skip-gram model:
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super(SkipGramModel, self).__init__()
        # Initialize an embedding layer that maps words to vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)        

    # For each input word, pass its vector through a fully connected layer with output size equal to the vocabulary size
    def forward(self, input_word: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(input_word)    # (batch_size, embedding_dim)
        out = self.linear(embeds)               # (batch_size, vocab_size)
        # Apply softmax activation to obtain probability distributions over the vocabulary for each training loop
        log_probs = F.log_softmax(out, dim=1)   # (batch_size, vocab_size)
        return log_probs

# Custom dataset for handling training data
class Word2VecDataset(Dataset):
    def __init__(self, training_data: List[Tuple[int, int]]) -> None:
        self.training_data = training_data
    
    def __len__(self) -> int:
        return len(self.training_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_word, context_word = self.training_data[idx]
        # Use tensors here for CUDA compatibility
        return torch.tensor(input_word, dtype=torch.long), torch.tensor(context_word, dtype=torch.long)

# Create a DataLoader for batching and shuffling data
def create_data_loader(training_data: List[Tuple[int, int]], batch_size: int) -> DataLoader:
    dataset = Word2VecDataset(training_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the training loop
def train_model(model: nn.Module, data_loader: DataLoader, epochs: int, learning_rate: float) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Move model to GPU if available
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        for input_words, target_words in data_loader:
            # Move input data to the same device as the model
            input_words, target_words = input_words.to(device), target_words.to(device)
            model.zero_grad()
            log_probs = model(input_words)
            loss = criterion(log_probs, target_words)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')

def save_model(model: nn.Module, filepath: str) -> None:
    """
    Save the model to a file.
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(mode_type:str, vocab_size:int, embedding_dim:int, filepath:str) -> nn.Module:
    """
    Load a saved model.
    """
    if model_type == "cbow":
        model = CBOWModel(vocab_size, embedding_dim)
    elif model_type == "skipgram":
        model = SkipGramModel(vocab_size, embedding_dim)
    else:
        raise ValueError("Invalid model type")
    
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    return model

def extract_embeddings(model: nn.Module) -> np.ndarray:
    """
    Extract word embeddings from the model.
    """
    return model.embeddings.weight.detach().cpu().numpy()

def find_similar_words(word: str, vocab: List[str], embeddings: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Find the top-k similar words to a given word.
    """
    if word not in vocab:
        raise ValueError(f"Word '{word}' not in vocabulary")
    
    word_idx = vocab.index(word)
    word_embedding = embeddings[word_idx]
    
    similarities = cosine_similarity([word_embedding], embeddings)[0]
    most_similar = similarities.argsort()[-top_k-1:-1][::-1]
    
    return [(vocab[idx], similarities[idx]) for idx in most_similar if vocab[idx] != word]

def visualize_embeddings(words: List[str], vocab: List[str], embeddings: np.ndarray) -> None:
    """
    Visualize word embeddings using t-SNE.
    """
    word_embeddings = [embeddings[vocab.index(word)] for word in words if word in vocab]
    
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(word_embeddings)
    
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        if word in vocab:
            x, y = embeddings_2d[i]
            plt.scatter(x, y)
            plt.annotate(word, (x, y))
    
    plt.title("t-SNE visualization of word embeddings")
    plt.show()

def analyze_analogy(word1: str, word2: str, word3: str, vocab: List[str], embeddings: np.ndarray) -> str:
    """
    Solve word analogies (e.g., "man" is to "woman" as "king" is to "queen").
    """
    if not all(word in vocab for word in [word1, word2, word3]):
        raise ValueError("All words must be in the vocabulary")
    
    vector = (embeddings[vocab.index(word2)] - embeddings[vocab.index(word1)] + 
              embeddings[vocab.index(word3)])
    
    similarities = cosine_similarity([vector], embeddings)[0]
    most_similar = similarities.argsort()[-1]
    
    return vocab[most_similar]

# Usage
if __name__ == "__main__":
    
    # Download and preprocess the Text8 dataset
    text8_path = download_text8()
    if text8_path is None:
        print("Failed to download or extract the dataset. Exiting.")
        exit(1)

    token_indices, vocab = preprocess_text8(text8_path)
    if token_indices is None or vocab is None:
        print("Failed to preprocess the dataset. Exiting.")
        exit(1)
    
    # Create training data
    training_data = create_training_data(token_indices)

    # Constants
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 100
    BATCH_SIZE = 256
    EPOCHS = 5
    LEARNING_RATE = 0.01

    # Create DataLoader
    data_loader = create_data_loader(training_data, BATCH_SIZE)
    
    # Model selection
    model_type = input("Choose model type (cbow/skipgram): ").lower()
    if model_type == "cbow":
        model = CBOWModel(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)
    elif model_type == "skipgram":
        model = SkipGramModel(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)
    else:
        print("Invalid model type. Exiting.")
        exit(1)

    # Compile the model for potential speedup (PyTorch 2.0+ feature)
    model = torch.compile(model)

    # Train the model
    train_model(model, data_loader, EPOCHS, LEARNING_RATE)

    print("Training completed.")

    # Save the model
    save_path = f"{model_type}_model.pth"
    save_model(model, save_path)

     # After training and saving the model:
    model_type = "cbow"  # or "skipgram", depending on what you chose earlier
    save_path = f"{model_type}_model.pth"

    # Load the model
    loaded_model = load_model(model_type, VOCAB_SIZE, EMBEDDING_DIM, save_path)

    # Extract embeddings
    embeddings = extract_embeddings(loaded_model)

    # Find similar words
    word = "king"
    similar_words = find_similar_words(word, vocab, embeddings)
    print(f"Words similar to '{word}': {similar_words}")

    # Visualize embeddings
    words_to_visualize = ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl"]
    visualize_embeddings(words_to_visualize, vocab, embeddings)

    # Analyze analogy
    word1, word2, word3 = "man", "woman", "king"
    analogy_result = analyze_analogy(word1, word2, word3, vocab, embeddings)
    print(f"'{word1}' is to '{word2}' as '{word3}' is to '{analogy_result}'")