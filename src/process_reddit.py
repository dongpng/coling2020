import lzma
import json
import csv
import os

import config
import util 

import DataWrapper


def analyze_social_patterns(input_file):
    """ Analyze social patterns in Reddit 
    - input_file: input file with variants
    """
    # First read in the spelling variants
    # standard, non standard
    # us, uk spelling
    standard_non_standard_pairs = util.read_variants_from_file(input_file)
    all_forms = set()
    all_forms = all_forms.union(standard_non_standard_pairs.keys())
    all_forms = all_forms.union(standard_non_standard_pairs.values())
    
    # Keeping track of stats
    subreddit_count_standard = {}
    subreddit_count_non_standard = {}

    # Data
    sentences_metadata = DataWrapper.DataWrapperGzipMulti(config.REDDIT_PROCESSED_DATA,
                                                          config.REDDIT_METADATA_DATA)
   
    for sentence, metadata in sentences_metadata:

        author, author_text, subreddit, link = metadata.split("\t")

        # Now check for tokens in our list
        for token in sentence:
            if token in all_forms:

                if subreddit not in subreddit_count_standard:
                    subreddit_count_standard[subreddit] = 0
                    subreddit_count_non_standard[subreddit] = 0

                # standard or US spelling
                if token in standard_non_standard_pairs:
                    subreddit_count_standard[subreddit] += 1
                else:
                    subreddit_count_non_standard[subreddit] += 1

    # Print out stats
    output_file_name = os.path.splitext(os.path.basename(input_file))[0]
    with open(output_file_name + "_subreddits_distr.txt", "w") as output_file:
        for subreddit, count in subreddit_count_standard.items():
            output_file.write("%s\t%s\t%s\n" % (subreddit, count, subreddit_count_non_standard[subreddit]))


if __name__ == "__main__":
    for input_file in config.SPELLING_VARIANTS_FILES:
        analyze_social_patterns(input_file)
    