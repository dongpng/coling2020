import glob
import pandas as pd
    
import argparse
import codecs
import os
import random

from random import randint, choice, choices
import numpy as np

import config
import util
import EmbeddingWrapper



#### Code to run the analogy experiment ""

RUN_STANDARD = "standard"
RUN_BASELINE = "baseline"

input_files = [config.DATA_DIR + "analogies/lengthening_analogies.txt",
               config.DATA_DIR + "analogies/vowel_omission_analogies.txt",
                config.DATA_DIR + "analogies/swapped_analogies.txt",
                config.DATA_DIR + "analogies/keyboard_substitution_analogies.txt",
                config.DATA_DIR + "analogies/g_dropping_analogies.txt",
                config.DATA_DIR +"analogies/common_misspellings_analogies.txt", 
                config.DATA_DIR +"analogies/us_uk_analogies.txt"]



datasets = ["twitter", "reddit"] 

runs = [RUN_STANDARD, RUN_BASELINE]

#### Create the dataset ######


def create_analogy_datasets():
    """ Method to create the analogy datasets 
    Reads in the spelling variants and creates the analogies.
    """
    create_evaluation_datasets_lengthening(config.DATA_DIR + "spelling_variants/lengthening_pairs.txt", 
                               config.DATA_DIR + "analogies_generated/lengthening_analogies.txt", 10)

    create_evaluation_datasets(config.DATA_DIR + "spelling_variants/vowel_omission_pairs.txt", 
                               config.DATA_DIR + "analogies_generated/vowel_omission_analogies.txt", 10)
    
    create_evaluation_datasets(config.DATA_DIR + "spelling_variants/swapped_pairs.txt", 
                               config.DATA_DIR + "analogies_generated/swapped_analogies.txt", 10)
    
    create_evaluation_datasets(config.DATA_DIR + "spelling_variants/keyboard_substitution_pairs.txt", 
                               config.DATA_DIR + "analogies_generated/keyboard_substitution_analogies.txt", 10)

    create_evaluation_datasets(config.DATA_DIR + "spelling_variants/common_misspellings_pairs.txt", 
                               config.DATA_DIR + "analogies_generated/common_misspellings_analogies.txt", 10)

    create_evaluation_datasets(config.DATA_DIR + "spelling_variants/us_uk_pairs.txt", 
                               config.DATA_DIR + "analogies_generated/us_uk_analogies.txt", 10)
    
    create_evaluation_datasets(config.DATA_DIR + "spelling_variants/g_dropping_pairs.txt", 
                               config.DATA_DIR + "analogies_generated/g_dropping_analogies.txt", 10)
    return


def create_evaluation_datasets(input_file, output_file, num_pairs, random_seed=9072342):
    """ Read in an input file with variant pairs. Randomly pair up a pair with ten other pairs.
    Write results to file

    - input_file: path to input file with variant pairs
    - output_file: path to output file
    - number of pairs to sample for each pair
    """

    # First read in the data
    pairs = []
    for standard, non_standard in util.read_variants_from_file(input_file).items():
        pairs.append((non_standard, standard))

    # Write output
    random.seed(random_seed)
    with open(output_file, 'w') as output_file:

        for pair in pairs:
            pairs2 = list(pairs) #copy
            pairs2.remove(pair)
            random.shuffle(pairs2)
            selected_pairs = pairs2[:num_pairs]
            for pair2 in selected_pairs:
                output_file.write("%s\t%s\t%s\t%s\n" % 
                                 (pair[0], pair[1], pair2[0], pair2[1]))


def create_evaluation_datasets_lengthening(input_file, output_file, num_pairs, random_seed=72403):
    """ Read in an input file with variant pairs. 
        Randomly pair up a pair with ten other pairs.
        This is for the case of lengthening, where we try to pair 
        up variants with the same amount of lengthening
        Write results to file

    - input_file: path to input file with variant pairs
    - output_file: path to output file
    - number of pairs to sample for each pair
    """

    pairs = {}
    for standard, non_standard in util.read_variants_from_file(input_file).items():
        # calculate amount of lengthening
        amount_lengthening = len(non_standard) - len(standard)
        # store the pair
        if amount_lengthening not in pairs:
            pairs[amount_lengthening] = []
        pairs[amount_lengthening].append((non_standard, standard))

    random.seed(random_seed)
    with open(output_file, 'w') as output_file:

        for amount_lengthening, p in pairs.items():
            for pair in p:
                pairs2 = list(p) #copy
                pairs2.remove(pair)
                random.shuffle(pairs2)
                selected_pairs = pairs2[:num_pairs]

                for pair2 in selected_pairs:
                    output_file.write("%s\t%s\t%s\t%s\n" % 
                                      (pair[0], pair[1], pair2[0], pair2[1]))


def read_analogy_dataset(input_file):
    """ Read in an inputfile with the analogies. Return a list with analogy instances

    - input_file: path to input file
    """
    result = []
    with codecs.open(input_file, 'r', 'utf-8') as input_file:
        for line in input_file.readlines():
            nonstandard1, standard1, nonstandard2, standard2 = line.strip().split("\t")
            result.append((nonstandard1, standard1, nonstandard2, standard2))
    return result


#### Running the experiments ######

def run_experiment(input_files, datasets, runs, embedding_type="word2vec", 
                    cutoff_rank=50, num_dims=config.NUM_DIMS, 
                    output_dir_location="analogy_results"):
    """ Run experiment. Write results to file 
    
    input_files:  list of input files for experiments. 
                  Input format. Each instance on a seperate line. 
                  [nonstandard1] \t [standard1] \t [nonstandard2] \t [standard2]
                  
    datasets: list of datasets to run experiments for (twitter, reddit)
    runs: list of runs (e.g., RUN_STANDARD, RUN_BASELINE)
    embedding_type: the embedding type to run experiments for
    cutoff_rank: only these top x results are saved
    num_dims: list with embedding dimensions 
    output_dir_location: output directory for output results.
    """

    if not os.path.isdir(output_dir_location):
        print("Make output directory: %s" % output_dir_location)
        os.makedirs(output_dir_location)


    for input_file in input_files:
        for dataset in datasets:
            for run in runs:
                output_file_name = ("analogy_results_%s_%s_%s_%s.txt" % 
                                    (os.path.splitext(os.path.basename(input_file))[0], dataset, run, embedding_type))
                output_file_location = os.path.join(output_dir_location, output_file_name)
               
                with codecs.open(output_file_location, 'w', 'utf-8') as output_file:
                    output_file.write("Input file %s\n" % os.path.splitext(os.path.basename(input_file))[0])
                    output_file.write("Dataset %s\n" % dataset)
                    output_file.write("Run %s\n" % run)
                    output_file.write("Embedding type %s\n" % embedding_type)
                    output_file.write("Cutoff rank %s\n" % cutoff_rank)
                    
                    output_file.write("####\n")

                    path_input_file = input_file

                    analogy_data = read_analogy_dataset(path_input_file)
                    

                    # For each dimension
                    for num_dim in num_dims:

                        # Get the model
                        print("Model %s" % embedding_type)
                        print("Dataset %s" % dataset)
                        print("Num dim %s" % num_dim)
                        
                        embedding_model = EmbeddingWrapper.EmbeddingWrapper(embedding_type, dataset, num_dim)
    
                        
                        vocab = embedding_model.get_vocab()
                        num_words = cutoff_rank 

                        for row in analogy_data:
                            nonstandard1, standard1, nonstandard2, standard2 = row

                            if (nonstandard1 not in vocab or nonstandard2 not in vocab or 
                                standard1 not in vocab or standard2 not in vocab):
                                
                                print("ERROR %s %s %s %s" % (nonstandard1, (nonstandard1 in vocab),
                                                             nonstandard2, (nonstandard2 in vocab)))
                                continue

                            
                            # Get analogy
                            results = None
                            if run == RUN_STANDARD:
                                # in the case of fasttext, only results are returned up until right answer
                                results = embedding_model.get_analogy(standard2, nonstandard1, 
                                            nonstandard2, num_words, standard1)
                                    
                            elif run == RUN_BASELINE:
                                results = embedding_model.get_most_similar(nonstandard1, num_words)

                            # only keep the words
                            results_words = [word for word, score in results]
                            
                            # Write to file
                            output_file.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (num_dim, standard1, standard2, 
                                                nonstandard1, nonstandard2, " ".join(results_words)))
                            output_file.flush()
                                    
                       

def get_results(evaluation_file, print_header=True):
    """ Read in output file (generated by run_experiment()) and return evaluation statistics
    
    
    Test input (filename called analogy_results_in_ing_analogies_reddit_baseline_fasttext.txt):
    Input file spelling_variations/reddit_verbs_uk_us_spelling.txt
    Dataset reddit
    Run standard
    Embedding type word2vec
    Cutoff rank 5
    ####
    50	car	tree	apple	house	book table car chair man
    50	book	table	chair	money	book apple house bag door
    100	a	b	c	d	b test1 test2 test3 a

    Output should be:
    	accuracy	dataset	embedding_type	filename	num_dim	rr	run	total_pairs
    0	0.5	reddit	fasttext	in_ing_analogies	50	0.666666667	baseline	2
    1	0	reddit	fasttext	in_ing_analogies	100	0.2	baseline	1


    """
    result_map = {}

    # extract the dataset from the filename
    filename = os.path.splitext(os.path.basename(evaluation_file))[0]
    dataset = filename.split("_")[-3]
    run = filename.split("_")[-2]
    embedding_type = filename.split("_")[-1]
    analogy_file = "_".join(filename.split("_")[2:-3])

    with open(evaluation_file, 'r') as input_file:
       
        # skip the first few lines
        skip = True
        for line in input_file.readlines():
            if line.strip() == "####":
                skip = False
                continue

            if not skip:
                
                num_dim, standard1, _, _, _, result = line.strip().split("\t")
                
                # create structure to save statistics
                if num_dim not in result_map:
                    result_map[num_dim] = {'total': 0, 'correct': 0, 'rr': []}

                result_words = result.split(" ")

                # total count
                result_map[num_dim]['total'] += 1

                # accuracy
                if result_words[0] == standard1:
                    result_map[num_dim]['correct'] += 1

                # reciprocal rank
                reciprocal_rank = return_reciprocal_rank(result_words, standard1)
                result_map[num_dim]['rr'].append(reciprocal_rank)

               
                # to check the results (no overlap and right number returned)
                if len(result_words) != len(set(result_words)):
                    print("error: overlap")
                    print(line)
                    print(input_file)
                if reciprocal_rank == 0.0 and len(result_words) != 50:
                    print("error: num results")
                    print(line)
                    print(input_file)
           
    # Now print the results
    results = []
    for num_dim in result_map.keys():       
        results.append({"filename": analogy_file, 
                        "dataset": dataset, 
                        "run": run, 
                        "embedding_type": embedding_type, 
                        "num_dim": num_dim,
                        "total_pairs": result_map[num_dim]['total'], 
                        "accuracy": float(result_map[num_dim]['correct'])/result_map[num_dim]['total'], 
                        "rr": np.array(result_map[num_dim]['rr']).mean()})
    return results


def return_reciprocal_rank(results, correct_answer):
    """ the results returned by word2vec.

    Input:
    - results: the ordered list of words
    - the correct answer
    - the common vocab. Only take these in consideration.

    Output:
    - Return reciprocal rank"""
    found_rank = -1
    for rank, word in enumerate(results):
        #if word not in common_vocab:
        #    continue
        if word == correct_answer:
            found_rank = rank + 1
            break

    if found_rank == -1:
        return 0.0 #if we apply a threshold, just return 0
    else:   
        return 1.0/found_rank



def print_all_results(input_dir=config.DATA_DIR + "analogy_results"):
    """ Loop through all files in the directory and print the results to csv file
    input_dir: path to directory with the results
    """
    result = []
    for input_file in glob.glob(input_dir+ "/analogy_results_*.txt"):
        result.extend(get_results(input_file))
    
    df = pd.DataFrame(result)
    df.to_csv("merged_analogy_results.csv",
        columns = ["accuracy", "dataset", "embedding_type",
                	"filename", "num_dim", "rr", "run", "total_pairs"]
    )


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='Analogy experiments')

    # Add the arguments
    my_parser.add_argument('action',
                        metavar='action',
                        type=str,
                        help='Which action to perform: create_data, run_skipgram, run_fasttext, evaluate')

    my_parser.add_argument('--results_dir',
                        metavar='results_dir',
                        type=str,
                        help='path to results dir',
                        required=False)

    # Execute the parse_args() method
    args = my_parser.parse_args()
    args = vars(args)
    
    print("Analogy experiment")
    if args["action"] == "create_data":
        # Create analogy dataset
        print("Create the dataset")
        create_analogy_datasets()

    elif args["action"] == "run_skipgram":
        # Run skipgram analogies
        print("Skipgram analogies")

        if not args["results_dir"]:
            print("Please specify the directory to print the results")

        run_experiment(input_files, datasets, runs, 
                        embedding_type="word2vec",
                        output_dir_location=args["results_dir"])

    elif args["action"] == "run_fasttext":
        # Run fastText analogies
        print("fastText analogies")

        if not args["results_dir"]:
            print("Please specify the directory to print the results")
       
        run_experiment(input_files, datasets, runs, 
                        embedding_type="fasttext",
                        output_dir_location=args["results_dir"])
    
    elif args["action"] == "evaluate":
        # Processes the output files and writes out
        # a summary table
        print("Evaluate: %s" % args["results_dir"] )
        print_all_results(args["results_dir"] + "/") 

    
    