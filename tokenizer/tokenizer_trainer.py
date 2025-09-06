import json

'''
Your assignment is to implement BPE in the following method. You can add
classes or other routines if you find them helpful. 

This method should save two output files:
./vocab.txt : a list of the final vocabulary in order, one entry per line
./merges.json : a list of tuples of merges, in order

NOTE: Typically these file extensions are reversed (the vocabulary is a
json file and the merge list is a txt file), but for our purposes this way seems
simplier.

Does not need to return anything.

-------

This should implement a GPT-style tokenizer which prefixes words with a space.
You can assume that the base vocabulary contains all single characters that will occur.
Treat punctuation (besides spaces) just like the other characters.

You do NOT need to worry about using a placeholder token in place of a space. 
You do NOT need to worry about special tokens (pad, bos, eos, unk, etc.). We have not covered these yet.

IMPORTANT: If there are ties while computing the merges, you should use lexigraphic order to resolve.
Points will be taken off if a different tie-break is used as it will not match the homework solution.

For example, if the pairs ('ab','out') and ('sp','ite') are tied for most occuring,
then "about" should be recorded before "spite".
'''

import re
from collections import Counter
from saving import save_merges, save_vocab

def split_with_delimiter(string, delimiter):
    ''' This splits along the delimiter, but the result includes the delimiter in the split. '''
    return re.split(f"(?={re.escape(delimiter)})", string)


def split_along_vocab(text, subs):
    ''' This splits words into tokens defined in the vocab list. '''
    # Sort so that longer matches are tried first
    subs = sorted(subs, key=len, reverse=True)
    i = 0
    result = []
    while i < len(text):
        for sub in subs:
            if text.startswith(sub, i):
                result.append(sub)
                i += len(sub)
                break
        else:
            # No match so take single character
            result.append(text[i])
            i += 1
    return result

def choose_pair_to_merge(pair_counts):
    ''' This finds the highest counted pairs, and if there are many it breaks the tie alphabetically. '''
    best_count = max(pair_counts.values())
    candidates = [p for p,c in pair_counts.items() if c == best_count]
    best_pair = min(candidates, key=lambda p: p[0] + p[1]) 
    return best_pair

def merge(pair, vocabulary, merge_list):
    ''' This merges two tokens together. '''
    merged = pair[0] + pair[1]
    vocabulary.append(merged)
    merge_list.append(pair)


def train_tokenizer(txt_file, vocab_size, base_vocabulary):
    '''
    param : txt_file - a string path to a text file of data, i.e. "./data.txt"
    param : vocab_size - integer specifying the final vocab size
    param : base_vocabulary - list of strings to add to the vocabulary by default

    saves:
    ./vocab.txt : a list of the final vocabulary in order, one entry per line, ties broken alphabetically
    ./merges.json : a list of tuples of merges, in order
    '''

    # Open the file and load the corpus
    with open(txt_file, 'r') as file:
        corpus = file.read()
        merges = []

        # Build a word frequency dictionary
        words = split_with_delimiter(corpus, ' ')
        word_frequencies = Counter(words)
        
        # Build up until the vocab size
        while len(base_vocabulary) != vocab_size:
            # Now we iterate over word frequencies and count the pairs
            pair_counts = {}
            for word, freq in word_frequencies.items():
                tokens = split_along_vocab(word, base_vocabulary)
                
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq
                
            # Now that we counted all pairs in the corpus, we pick the most frequent
            most_frequent_pair = choose_pair_to_merge(pair_counts)
            
            # Merge the most frequent pair
            merge(most_frequent_pair, base_vocabulary, merges)
        
        # Save the vocab and merges once its all done
        save_vocab(base_vocabulary)
        save_merges(merges)

if __name__ == "__main__":

    # example of using this method.
    base = "abcdefghijklmnopqrstuvwxyz"
    base += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base += "0123456789"
    base += "!@#$%^&*()_+-=[]{}|;':,.<>/?`~ "
    base += "\\"
    base += '"'

    train_tokenizer("./data2.txt", len(base)+50, [c for c in base])