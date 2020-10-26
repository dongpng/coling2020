import configparser

import gensim
from gensim.models import KeyedVectors

# Main directory
config = configparser.ConfigParser()
config.read('config.ini')
MAIN_DIR = config['main']['main_dir']

EXPERIMENTS_DIR = MAIN_DIR + "experiments/"
DATA_DIR = MAIN_DIR + "data/"
EMBEDDINGS_DIR = MAIN_DIR + "embeddings/"

# Embedding parameters
NUM_DIMS = [50, 100, 150, 200, 250, 300]

# List of the files with spelling variants
SPELLING_VARIANTS_FILES =  [DATA_DIR + "spelling_variants/vowel_omission_pairs.txt",
                            DATA_DIR + "spelling_variants/swapped_pairs.txt",
                            DATA_DIR + "spelling_variants/keyboard_substitution_pairs.txt",
                            DATA_DIR + "spelling_variants/g_dropping_pairs.txt",
                            DATA_DIR + "spelling_variants/common_misspellings_pairs.txt", 
                            DATA_DIR + "spelling_variants/us_uk_pairs.txt",
                            DATA_DIR + "spelling_variants/lengthening_pairs.txt",
                            ]


# Data dirs
TWITTER_DATA_DIR = DATA_DIR + "twitter/"
REDDIT_DATA_DIR = DATA_DIR + "reddit/"

# Vocab files
REDDIT_VOCAB_FILE = REDDIT_DATA_DIR + "reddit_processed_2018_05_10_vocab_count.txt"
TWITTER_VOCAB_FILE = TWITTER_DATA_DIR + "london_tweets_processed_may2018_april2019_vocab_count.txt"

# Trained embeddings
TWITTER_EMBEDDINGS_DIR = EMBEDDINGS_DIR + "twitter/"
REDDIT_EMBEDDINGS_DIR = EMBEDDINGS_DIR + "reddit/"