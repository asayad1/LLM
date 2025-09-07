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
from datetime import datetime
from saving import save_merges, save_vocab

def split_with_delimiter(string, delimiter):
    ''' This splits along the delimiter, but the result includes the delimiter in the split. '''
    return re.split(f"(?={re.escape(delimiter)})", string)


def split_along_vocab(text, subs):
    ''' This splits words into tokens defined in the vocab list. '''
    # This is now expected to bring in a sorted (longest-first) vocab
    # I did this because previously I was sorting once per every unique word when instead it should be once per iteration
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
    candidates = [p for p, c in pair_counts.items() if c == best_count]
    best_pair = min(candidates, key=lambda p: p[0] + p[1]) 
    return best_pair

def merge(pair, vocabulary, merge_list):
    ''' This merges two tokens together. '''
    merged = pair[0] + pair[1]
    vocabulary.append(merged)
    merge_list.append(pair)

def apply_merge_to_tokens(tokens, pair, merged_pair):
    ''' This replaces every adjacent (a,b) with merged_pair.'''
    a, b = pair
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        if i + 1 < n and tokens[i] == a and tokens[i + 1] == b:
            out.append(merged_pair)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out

def train_tokenizer(txt_file, vocab_size, base_vocabulary):
    '''
    param : txt_file - a string path to a text file of data, i.e. "./data.txt"
    param : vocab_size - integer specifying the final vocab size
    param : base_vocabulary - list of strings to add to the vocabulary by default

    saves:
    ./vocab.txt : a list of the final vocabulary in order, one entry per line, ties broken alphabetically
    ./merges.json : a list of tuples of merges, in order
    '''
    log_file = open("train.log", "a", buffering=1)

    # Open the file and load the corpus
    with open(txt_file, 'r') as file:
        corpus = file.read()
        merges = []

        # Build a word frequency dictionary
        words = split_with_delimiter(corpus, ' ')
        word_frequencies = Counter(words)

        # Sort vocab by length
        vocab_sorted_by_len = tuple(sorted(base_vocabulary, key=len, reverse=True))
        tokenized_cache = { word: split_along_vocab(word, vocab_sorted_by_len) for word in word_frequencies }
        
        # Build up until the vocab size
        while len(base_vocabulary) != vocab_size:
            # Now we iterate over word frequencies and count the pairs
            pair_counts = {}

            for word, freq in word_frequencies.items():
                tokens = tokenized_cache[word]
                per_word_pairs = Counter(zip(tokens, tokens[1:]))
                
                for pair, count_in_word in per_word_pairs.items():
                    pair_counts[pair] = pair_counts.get(pair, 0) + count_in_word * freq

            # See if any more pairs exist (this is necessary otherwise theres a max on empty error)
            if not pair_counts:
                break

            # Now that we counted all pairs in the corpus, we pick the most frequent
            most_frequent_pair = choose_pair_to_merge(pair_counts)
            merged_tokens = most_frequent_pair[0] + most_frequent_pair[1] 

            # Merge the most frequent pair
            merge(most_frequent_pair, base_vocabulary, merges)

            # Rebuild longest-first vocab which includes the new symbol
            vocab_sorted_by_len = tuple(sorted(base_vocabulary, key=len, reverse=True))
            
            # Apply the merge to every cached token list
            for word, old_tokens in list(tokenized_cache.items()):
                # Apply in-place merge of the chosen pair
                new_tokens = apply_merge_to_tokens(old_tokens, most_frequent_pair, merged_tokens)

                if new_tokens != old_tokens:
                    # Re-tokenize from the original word with updated vocab
                    tokenized_cache[word] = split_along_vocab(word, vocab_sorted_by_len)
                else:
                    tokenized_cache[word] = old_tokens

            # Print training progress
            print(f'{(len(base_vocabulary) / vocab_size) :.3%} '
                f'({len(base_vocabulary)} / {vocab_size} learned vocab) - '
                f'{datetime.now().strftime("%H:%M:%S")}',
                file=log_file, flush=False)
        
        # Save the vocab and merges once its all done
        save_vocab(base_vocabulary)
        save_merges(merges)
    
    log_file.close()


if __name__ == "__main__":
    # Example of using this method.
    base = "abcdefghijklmnopqrstuvwxyz"
    base += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base += "0123456789"
    base += "!@#$%^&*()_+-=[]{}|;':,.<>/?`~ "
    base += "\\"
    base += '"'

    train_tokenizer("./data.txt", len(base)+50000, [c for c in base])