import argparse

import os
import math
import glob

import numpy as np

from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.metrics import classification_report,f1_score
from sklearn.model_selection import KFold
from scipy.spatial import distance


import config
import DataWrapper
import EmbeddingWrapper
import util

#PATTERN_INDICATOR = "_PAT#_" # separates the label from a pattern used to split the data
VECTOR_TYPE_DIFFERENCE = "difference_vector"
VECTOR_TYPE_NORM_DIFFERENCE = "norm_difference_vector"
VECTOR_TYPE_STANDARD = "standard_vector"
VECTOR_TYPE_NON_STANDARD = "non_standard_vector"


#############  Classification utilities ############# 
def create_classification_dataset(input_files, output_file="classification_dataset.txt"):
    """ Aggregate across analogy files to create a dataset for classification """
    label_map = {}
    dataset = {}
    with open(output_file, 'w') as output_file:
        for i, input_file in enumerate(input_files):
            label_map[input_file] = i
            variants = util.read_variants_from_file(input_file)
            for standard, non_standard in variants.items():
                # this way each spelling variation type gets a different number
                dataset[non_standard] = (i, standard)

    
        for word, data in dataset.items():
            label, standard = data
            output_file.write("%s\t%s\t%s\n" % (word, label, standard))

    print(label_map)


def read_data_aux_task(input_file):
    """ Read in the data in the format: word [tab] label [tab] standard
    Label is assumed to be int
    """
    result = {}
    with open(input_file, 'r') as input_file:
        for line in input_file.readlines():
            word, label, standard = line.strip().split("\t")
            result[word] = (int(label), standard)

    return result




def write_data_auxiliary_tasks(model, dataset, input_file, 
                               output_dir,
                               include_meta_data=False, 
                               include_embeddings=True, 
                               normalized_embeddings=True,
                               vector_type=VECTOR_TYPE_DIFFERENCE,
                               num_dim = 300):
    """ Print out features for the auxiliary tasks
    -model: the embedding model
    -dataset: reddit or twitter
    -input_file: path to input file
    -output_dir: output directory
    -include_meta_data: whether to include metadata (freq)
    -include_embeddings: whether to include embeddings
    -normalized_embeddings: whether to include normalized embeddings
    -vector_type: if VECTOR_TYPE_DIFFERENCE, write the diff. between non-standard and standard vector
                  if VECTOR_TYPE_STANDARD, with the standard vector
    """

    if not os.path.isdir(output_dir):
        print("Make output directory: %s" % output_dir)
        os.makedirs(output_dir)

    # Get vocab count
    vocab_count = None
    if dataset == "twitter":
        vocab_count = util.read_vocab_count(config.TWITTER_VOCAB_FILE)
    else:
        vocab_count = util.read_vocab_count(config.REDDIT_VOCAB_FILE)
    
    # read in the right model
    f = EmbeddingWrapper.EmbeddingWrapper(model, dataset, num_dim)
    
    # The data that we want to use
    data_aux_task = read_data_aux_task(input_file)

    # write data to file. The few lines below are for 
    # determining the filename
    metadata = "_metadata" if include_meta_data else ""
    embeddings = "_embeddings" if include_embeddings else ""
    normalized = "_normalized" if normalized_embeddings else ""

    with open(output_dir + 'classification_data_' + str(num_dim) + metadata + embeddings + '_' + model + '_' + 
                dataset + '_' + vector_type + normalized + '.txt', 'w') as output_file:
        
        for word, d in data_aux_task.items():

            # Get input data
            label, standard = d
            vector = ""

            # Get the embeddings of the standard and non-standard forms
            vector_non_standard, vector_standard = None,None

            if normalized_embeddings:
                vector_non_standard = f.get_norm_embedding(word)  
                vector_standard = f.get_norm_embedding(standard) 
            else:
                vector_non_standard = f.get_embedding(word)  
                vector_standard = f.get_embedding(standard) 

            # Now determine the vector to output
            if include_embeddings:
                if vector_type == VECTOR_TYPE_DIFFERENCE:
                    vector = vector_non_standard - vector_standard
                elif vector_type == VECTOR_TYPE_NORM_DIFFERENCE:
                    vector = vector_non_standard - vector_standard
                    vector /= np.linalg.norm(vector.astype(float))
                elif vector_type == VECTOR_TYPE_STANDARD:
                    vector = vector_standard
                elif vector_type == VECTOR_TYPE_NON_STANDARD:
                    vector = vector_non_standard
                else:
                    print("Vector type unknown")
                
                vector =  "\t" + "\t".join(["%.10f" % v for v in vector]) 
 
            if include_meta_data:
                vector += "\t" + str(math.log(vocab_count[word], 10))
            
            vector += "\t" + str(label)

            output_file.write(word + "\t" + vector.strip() + "\n")


######Data utilities  ######
def read_X_Y(input_file):
    """Read in the input file. 
    assuming the following format (tab seperated):
    0: word
    .....: x values
    y: the label (int)

    """
    # init
    X = []
    Y = []
    words = [] # the words corresponding to the instances

    # Read the data
    with open(input_file, 'r') as input_file:
        for line in input_file.readlines():
            values = line.split("\t")

            # read in the words
            words.append(values[0])

            # read in x values
            tmp_x = [float(x) for x in values[1:-1]]
            X.append(np.array(tmp_x))

            # read in y value
            tmp_y = int(values[-1])
            Y.append(tmp_y)
        
    X = np.array(X)
    Y = np.array(Y)

    return X, Y, words



def run_classification_helper(input_file, output_file):
    """ Run a classification experiment for one input file.
    Returns F1 micro, F1 macro
    input_file: path to input file
    output_file: path to output file
    """
    print("Input file %s" % input_file)

    # Read in the data
    X,Y,words = read_X_Y(input_file)
 
    # We use k-fold validation
    cv = KFold(n_splits=10, shuffle=True, random_state=8792342)

    predictions_all = []
    labels_all = []
    words_all = []

    for trainidx, testidx in cv.split(X, Y): 

        # Train logistic regression
        clf = LogisticRegression(max_iter=250)
        clf.fit(X[trainidx], Y[trainidx]) 

        # Get predictions
        predictions = clf.predict(X[testidx])
        
        # Save the data
        predictions_all.extend(predictions)
        labels_all.extend(list(Y[testidx]))
        words_all.extend(np.array(words)[testidx])
       
        
    # Print the classification results
    print_classification_evaluation(input_file, predictions_all, labels_all, words_all, output_file) 

    return (f1_score(labels_all, predictions_all, average='micro'),
            f1_score(labels_all, predictions_all, average='macro'))


def print_classification_evaluation(input_file, predictions, groundtruth, words, output_file_path):
    """Evaluate and print out evaluation results, including predictions per word
    
    input_file: path to input file for classification
    predictions: list of predictions
    groundtruth: list of groundtruth labels
    word: list of words
    output_file: file to write predictions to
    
     """
    from sklearn.metrics import confusion_matrix

    with open(output_file_path, 'w') as output_file:
        # General classification report from sklearn
        output_file.write("Input file %s\n" % input_file)
        output_file.write("Sklearn classification report\n")
        output_file.write(classification_report(groundtruth, predictions) + "\n\n")
        output_file.write("Confusion matrix (rows=ground truth)\n")
        output_file.write(str(confusion_matrix(groundtruth, predictions)))

        output_file.write("\n\nPredictions\n\n")
        output_file.write("word\tpred\tground truth\n")
        for i, pred in enumerate(predictions):
            output_file.write("%s\t%s\t%s\n" % (words[i], pred, groundtruth[i]))


def run_majority_classifier():
    """Return a majority classifier"""
    data_aux_task = read_data_aux_task(config.DATA_DIR + "/auxiliary_tasks/classification_dataset.txt",)

    ground_truth = []
    for word, d in data_aux_task.items():
        label, standard = d
        ground_truth.append(label)
        
    # lengthening is the  majority category
    predictions = [5] * len(ground_truth)

    print("Sklearn classification report\n")
    print(classification_report(ground_truth, predictions) + "\n\n")
    print("micro %s" % f1_score(ground_truth, predictions, average='micro'))
    print("macro %s" % f1_score(ground_truth, predictions, average='macro'))

       
def run_classification(input_dir):
    """ Run classification experiments. Write output to classification_results.txt
    All files in input_dir will be used as input for classification experiments.
    
    """
    with open(input_dir + 'classification_results.txt', 'w') as global_output_file:
        global_output_file.write("input_file\tmodel\tdataset\trun\tf1 micro\tf1 macro\n")
        for input_file in glob.glob(input_dir + "classification_data_*.txt"):
            if input_file.endswith("_output.txt"):
                continue

            # Get path for output file
            output_file = os.path.splitext(input_file)[0] + "_output.txt"

            # Run classification and get micro f1 and macro f1
            micro_f1, macro_f1 = run_classification_helper(input_file, output_file)
            
            # Extract properties from input file name
            model = ""
            if "word2vec" in input_file:
                model = "word2vec"
            elif "fasttext" in input_file:
                model = "fasttext"

            dataset = ""
            if "reddit" in input_file:
                dataset = "reddit"
            elif "twitter" in input_file:
                dataset = "twitter"

            run = input_file[input_file.find(dataset) + len(dataset) + 1:]

            # Write results to file
            global_output_file.write("%s\t%s\t%s\t%s\t%.2f\t%.2f\n" % (os.path.split(input_file)[1], 
                                                                   model, dataset, run, 
                                                                   micro_f1, macro_f1))
            



if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='Diagnostic classifier experiments')

    # Add the arguments
    my_parser.add_argument('action',
                        metavar='action',
                        type=str,
                        help='Which action to perform: create_data, create_data_control1, create_data_control2,' + 
                             'create_data_diff, run_control1, run_diff')

    # Execute the parse_args() method
    args = my_parser.parse_args()
    args = vars(args)
    
    if args["action"] == "create_data":
        print("Create data")
        # Create the classificiation dataset
        spelling_variant_files =    [config.DATA_DIR + "spelling_variants/us_uk_pairs.txt",
                                    config.DATA_DIR + "spelling_variants/g_dropping_pairs.txt",
                                    config.DATA_DIR + "spelling_variants/swapped_pairs.txt",
                                    config.DATA_DIR + "spelling_variants/vowel_omission_pairs.txt",
                                    config.DATA_DIR + "spelling_variants/keyboard_substitution_pairs.txt",
                                    config.DATA_DIR + "spelling_variants/lengthening_pairs.txt",
                                    config.DATA_DIR + "spelling_variants/common_misspellings_pairs.txt"]


        create_classification_dataset(spelling_variant_files)
    

    # Running the actual experiments
    datasets = ["twitter", "reddit"]
    models = ["word2vec", "fasttext"]
    normalized = [True, False]
    for d in datasets:
        for m in models:
            for n in normalized:

                if args["action"] == "create_data_control1": 
                    print("Write data for control1")
                    write_data_auxiliary_tasks(m, d, config.DATA_DIR + 
                                                "auxiliary_tasks/classification_dataset.txt", 
                                                "output/control-standard-vector/",
                                                include_meta_data=False, 
                                                include_embeddings=True, 
                                                normalized_embeddings=n,
                                                vector_type=VECTOR_TYPE_STANDARD, 
                                                num_dim = 300)

                elif args["action"] == "create_data_control2":
                    print("Write data for control2")
                    write_data_auxiliary_tasks(m, d, config.DATA_DIR + 
                                                "/auxiliary_tasks/classification_dataset.txt", 
                                                "output/control-standard-vector-freq/",
                                                include_meta_data=True, 
                                                include_embeddings=True, 
                                                normalized_embeddings=n,
                                                vector_type=VECTOR_TYPE_STANDARD, 
                                                num_dim = 300)

                elif args["action"] == "create_data_diff":
                    print("Write data for difference vectors")
                    write_data_auxiliary_tasks(m, d, config.DATA_DIR +
                                                 "/auxiliary_tasks/classification_dataset.txt", 
                                                 "output/difference-vector/",
                                                include_meta_data=False, 
                                                include_embeddings=True, 
                                                normalized_embeddings=n,
                                                vector_type=VECTOR_TYPE_DIFFERENCE, 
                                                num_dim = 300)


    if args["action"] == "run_control1":
        run_classification("output/control-standard-vector/")
    elif args["action"] == "run_control2":
        run_classification("output/control-standard-vector-freq/")
    elif args["action"] == "run_diff":
        run_classification("output/difference-vector/")
    elif args["action"] == "run_majority":
        run_majority_classifier()
    

