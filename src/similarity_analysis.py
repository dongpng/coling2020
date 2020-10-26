import os
import random
import numpy as np
import pandas as pd
from pathlib import Path 

import config
import util
import EmbeddingWrapper

from scipy.spatial.distance import cosine

import numpy.testing as npt

def print_similarity_analysis(input_files, output_file, num_dim=300):
    """ Calculate cosine similarity between standard and non-standard variant 
    input_files: a list with files containing variant pairs
    """

    if not os.path.isdir(os.path.dirname(output_file)):
        print("Make output directory: %s" % os.path.dirname(output_file))
        os.mkdir(os.path.dirname(output_file))

    if os.path.isfile(output_file):
        print("Error: output file already exists")
        return

    # dataset
    datasets = ["twitter", "reddit"]
    embedding_models = ["fasttext", "word2vec"]

    results = []

    for dataset in datasets:

        # Read in vocab count to control for frequency
        vocab_count = None
        if dataset == "twitter":
            vocab_count = util.read_vocab_count(config.TWITTER_VOCAB_FILE)
        elif dataset == "reddit":
            vocab_count = util.read_vocab_count(config.REDDIT_VOCAB_FILE)

        # Read in the embedding models
        for embedding_model in embedding_models:

            print("%s\t%s\t%s" % (embedding_model, dataset, num_dim))
            model = EmbeddingWrapper.EmbeddingWrapper(embedding_model, dataset, num_dim)

            for input_file_path in input_files:
            
                # Read in the variants
                variants = []
                with open(input_file_path, 'r') as input_file:
                    for line in input_file.readlines():
                        non_standard, standard = line.strip().split("\t")[:2]
                        variants.append((non_standard, standard))


                input_file_name = Path(input_file_path).name

                # Write the data for each pair
                for non_standard, standard in variants:

                    if standard not in model.get_vocab() or non_standard not in model.get_vocab():
                        # should only happen with external datasets
                        print("Skip %s %s" % (non_standard, standard))
                        continue

                    s1 = model.get_norm_embedding(standard)
                    s2 = model.get_norm_embedding(non_standard)
                    
                    cosine_sim =  1-cosine(s1, s2)
                    result = {
                        "dataset": dataset,
                        "embedding_model": embedding_model,
                        "input_file": input_file_name,
                        "num_dim": num_dim,
                        "cosine_sim": cosine_sim,
                        "vocab_count_standard": vocab_count[standard],
                        "vocab_count_non_standard": vocab_count[non_standard],
                        "standard": standard,
                        "non-standard": non_standard
                    }
                    results.append(result)
                    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


def freq_to_words(vocab_count, only_these_words):
    """ Return a dictionary with freq to words
    vocab_count: mapping from word to count
    only_these_words: set of words, only keep these
    """
    result = {}
    for word, count in vocab_count.items():
        if word in only_these_words:
            if count not in result:
                result[count] = []
            result[count].append(word)
    
    return result


def sample_from_list(vocab_count_reversed, target_freq, 
                    words_ordered_by_freq, word_to_ignore):
    """ Sample a word with similar freq to target_freq, but don't return word_to_ignore 

    vocab_count_reversed: mapping from freq to words
    target_freq: the target freq. to sample word for
    words_ordered_by_freq: list of words ordered by freq
    word_to_ignore: the word to ignore
    """
    random.seed(target_freq + len(word_to_ignore))

    # easiest case, just use other words with the same target freq.
    candidates = vocab_count_reversed[target_freq]

    # if the only word with the frequency is the target word, look a bit further 
    if len(candidates) == 1:
        current_index = words_ordered_by_freq.index(word_to_ignore)
        candidates = words_ordered_by_freq[max(0, current_index-10): current_index + 10]
        
    # remove the target word from the candidates
    candidates = list(candidates) # copy to be sure
    candidates.remove(word_to_ignore)
   
    # now, choose a random candidate
    return random.choice(candidates)


def create_random_pairs(input_files, output_file="random_candidates.txt", min_freq_count=20):
    """ Create random pairs for sanity check, 
    We start with the original spelling variant pairs,
    and for each word we randomly match a word by frequency. 
    Final pairs need to occur in both datasets.
    Output: file with random pairs.

    input_files: pahts to the original spelling variant files
    min_freq_count: the minimum freq count
    """

    # read in the vocab counts
    vocab_count_twitter = util.read_vocab_count(config.TWITTER_VOCAB_FILE)
    vocab_count_reddit = util.read_vocab_count(config.REDDIT_VOCAB_FILE)

    # get common words (that meet freq threshold)
    common_vocab = set()
    for word, count in vocab_count_reddit.items():
        if count >= (min_freq_count and word in vocab_count_twitter and 
                    vocab_count_twitter[word] >= min_freq_count):
            common_vocab.add(word)
    
    # reverse vocab counts
    # create vocab with <freq, words>
    vocab_count_twitter_reversed = freq_to_words(vocab_count_twitter, common_vocab)
    vocab_count_reddit_reversed = freq_to_words(vocab_count_reddit, common_vocab)

    # words_ordered_by_freq
    twitter_words_ordered_by_freq = list(sorted(vocab_count_twitter, key=vocab_count_twitter.get, reverse=True))
    twitter_words_ordered_by_freq = [x for x in twitter_words_ordered_by_freq if x in common_vocab]
    reddit_words_ordered_by_freq = list(sorted(vocab_count_reddit, key=vocab_count_reddit.get, reverse=True))
    reddit_words_ordered_by_freq = [x for x in reddit_words_ordered_by_freq if x in common_vocab]
  
    # To switch between reddit and twitter
    num_processed = 0

    with open(output_file, 'w', encoding='utf-8') as output_file:
        for input_file_path in input_files: 
            with open(input_file_path, 'r') as input_file:
                for line in input_file.readlines():
                    non_standard, standard = line.strip().split("\t")[:2]
                    # switch between using twitter or reddit as base to sample
                    if num_processed % 2 == 0:
                        # twitter
                        freq_ns = vocab_count_twitter[non_standard]
                        freq_s = vocab_count_twitter[standard]

                      
                        w1_ns_freq = sample_from_list(vocab_count_twitter_reversed, freq_ns, 
                                                     twitter_words_ordered_by_freq, non_standard)

                        
                        w2_s_freq = sample_from_list(vocab_count_twitter_reversed, freq_s, 
                                                     twitter_words_ordered_by_freq, standard)

                       
                        output_file.write("%s\t%s\t%s\t%s\t%s\n" % (w1_ns_freq, w2_s_freq, 
                                                                    non_standard, standard, 'twitter'))
                    else:
                        # reddit
                        freq_ns = vocab_count_reddit[non_standard]
                        freq_s = vocab_count_reddit[standard]

                       
                        w1_ns_freq = sample_from_list(vocab_count_reddit_reversed, freq_ns, 
                                                     reddit_words_ordered_by_freq, non_standard)

                        
                        w2_s_freq = sample_from_list(vocab_count_reddit_reversed, freq_s, 
                                                     reddit_words_ordered_by_freq, standard)
                        
                        output_file.write("%s\t%s\t%s\t%s\t%s\n" % (w1_ns_freq, w2_s_freq, 
                                                                    non_standard, standard, 'reddit'))
                        
                    num_processed+=1


if __name__ == "__main__":
    # Create random pairs
    #create_random_pairs(config.SPELLING_VARIANTS_FILES, output_file="random_candidates.txt")

    # Run the analysis.
    print("Similarity analysis: spelling variants")
    print_similarity_analysis(config.SPELLING_VARIANTS_FILES, "output/cosine_similarities.csv")

    print("Similarity analysis: Random")
    print_similarity_analysis([config.DATA_DIR + "similarity_analysis/random_candidates.txt"], 
                             "output/cosine_similarities_random.csv")

    print("Similarity analysis: BATS")
    input_files_bats = [
         config.DATA_DIR + "BATS_3.0/1_Inflectional_morphology/I01 [noun - plural_reg].txt",
         config.DATA_DIR + "BATS_3.0/1_Inflectional_morphology/I06 [verb_inf - Ving].txt"
    ]

    print_similarity_analysis(input_files_bats, "output/cosine_similarities_bats.csv")
   