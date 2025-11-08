import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    
    # Tokenize the text into words
    words = word_tokenize(text)
    transformed_words = []
    
    # QWERTY keyboard layout for typo simulation
    keyboard_neighbors = {
        'a': ['q', 's', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'],
        'e': ['w', 'r', 'd', 's'],
        'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'],
        'h': ['g', 'y', 'u', 'j', 'n', 'b'],
        'i': ['u', 'o', 'k', 'j'],
        'j': ['h', 'u', 'i', 'k', 'm', 'n'],
        'k': ['j', 'i', 'o', 'l', 'm'],
        'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'p', 'l', 'k'],
        'p': ['o', 'l'],
        'q': ['w', 'a'],
        'r': ['e', 't', 'f', 'd'],
        's': ['a', 'w', 'e', 'd', 'x', 'z'],
        't': ['r', 'y', 'g', 'f'],
        'u': ['y', 'i', 'j', 'h'],
        'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'e', 's', 'a'],
        'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'u', 'h', 'g'],
        'z': ['a', 's', 'x']
    }
    
    for word in words:
        # Apply transformations with certain probabilities
        rand_val = random.random()
        
        # 30% chance of synonym replacement (if word is not punctuation and has synonyms)
        if rand_val < 0.3 and word.isalpha() and len(word) > 2:
            synsets = wordnet.synsets(word.lower())
            if synsets:
                # Get all lemmas from all synsets
                lemmas = []
                for syn in synsets[:3]:  # Only consider first 3 synsets
                    for lemma in syn.lemmas():
                        lemma_name = lemma.name().replace('_', ' ')
                        if lemma_name.lower() != word.lower():
                            lemmas.append(lemma_name)
                
                if lemmas:
                    # Choose a random synonym
                    synonym = random.choice(lemmas)
                    # Preserve original capitalization
                    if word[0].isupper():
                        synonym = synonym.capitalize()
                    transformed_words.append(synonym)
                    continue
        
        # 20% chance of introducing a typo (if word is alphabetic and longer than 2 chars)
        elif rand_val < 0.5 and word.isalpha() and len(word) > 2:
            word_list = list(word)
            # Choose a random position (not first or last character)
            if len(word) > 3:
                pos = random.randint(1, len(word) - 2)
                char = word_list[pos].lower()
                
                # Replace with a neighboring key
                if char in keyboard_neighbors:
                    replacement = random.choice(keyboard_neighbors[char])
                    # Preserve capitalization
                    if word_list[pos].isupper():
                        replacement = replacement.upper()
                    word_list[pos] = replacement
            
            transformed_words.append(''.join(word_list))
            continue
        
        # Keep the word unchanged
        transformed_words.append(word)
    
    # Detokenize the words back into text
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_words)

    ##### YOUR CODE ENDS HERE ######

    return example
