import numpy as np
import codecs

import config

SPECIAL_TOKENS = set(['@user', '[link]', '[num]'])

def read_variants_from_file(input_file, reversed=False):
    """Given a file with variants (non_standard \t standard) return a map
    input_file: path to input file
    reversed: if True, return a mapping from non-standard to standard
    """
    variants = {}
    with open(input_file, 'r') as input_file:
        for line in input_file.readlines():
            non_standard, standard = line.strip().split("\t")[:2] 
            if reversed:
                variants[non_standard] = standard
            else:
                variants[standard] = non_standard

    return variants



def read_vocab_count(vocab_count_file):
    """ Read in the vocab count file and return a map with word, count
    vocab_count_file: path to vocab count file
     """
    vocab_count = {}
    with codecs.open(vocab_count_file, 'r', 'utf-8') as input_file:
        for line in input_file.readlines():
            word, count = line.strip().split("\t")
            vocab_count[word] = int(count)

    return vocab_count
