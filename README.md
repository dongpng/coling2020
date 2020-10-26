# "Do word embeddings capture spelling variation?" Nguyen and Grieve, COLING 2020


## Installation

The following installes the packages into a conda environment.


**1)** The following creates a conda environment called *embeddings_spelling*

```
conda create -n embeddings_spelling python=3.7.6
```

Then:

```
conda install scikit-learn=0.22.1
conda install -c conda-forge gensim=3.8.1
conda install pandas
```

Also build fastText (recommended: follow instructions on https://fasttext.cc/docs/en/support.html)

```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .
```

The code was originally written with fastText 0.8, but has now also been tested with 0.9.2


For plotting the introductory figure:

```
conda install -c conda-forge jupyterlab
conda install ipykernel
pip install matplotlib
```

The full environment can be found in `environment.yml`

### File structure

More info about the individual files is provided at the bottom of this README.

**src/config.ini**: add the path to the main directory here. So, the path referring to the directory with the following structure:

**data**: only the spelling variants list is provided in this Github repositiory. For all other files, see the Zenodo repo (LINK TO BE ADDED).

.
├──data
├──src
├──README.md
├──environment.yml
├──environment_from_history.yml



## Data

**Embeddings**

The directories contain embeddings trained using both fastText and word2vec (gensim).

- Reddit embeddings: `data/embeddings/reddit/`
- Twitter embeddings: `data/embeddings/twitter/`

**Datasets**

The raw Twitter and Reddit datasets are not provided.
Statistics are provided.

- Reddit data: `data/reddit`: Contains the vocab count files (word, freq. count) and the used subreddits
- Twitter: `data/twitter`: The vocab count file


**Spelling variants**
This paper looks at seven different types of spelling variations:

- Lengthening
- Swapped characters 
- Common misspellings
- G-dropping
- Vowel omission
- Nearby character substitution
- British vs. American spelling


The lists with spelling variants can be found in `data/spelling_variants`.
Each file contains pairs with conventional and non-conventional forms.

*Format:*

`[non-conventional form] [tab] [conventional form]`

## Embeddings

**Parameters:**

The embeddings are trained with the following parameters:

* Dimension size: 50, 100, 150, 200, 250, 300
* Context-window:  5
* min-count: 20

Number of iterations/epochs is 5, which is standard for both gensim and the fasttext library.

The full data files for training the embeddings are not provided but the code is, to document the used parameters and methods.

**Word2vec (skipgram)**

Trained using: `gensim_word2vec.py` (skipgram)

**FastText**

Trained using `train-embeddings.sh` (skipgram

## Extra linguistic variation

### Twitter:

This was done using the m3 tool ```https://github.com/euagendas/m3inference```, Wang et al. 2019.

Steps:

1. For each spelling, we sample 5 Twitters users from our data.
2. We then apply the M3 inference tool. For some users the tool wasn't able to provide an estimate. The output is in `data/social_variation_analysis/twitter/`. 
3. Analyze them using `social_variation.rmd`


### Reddit

The statistics were created by running `process_reddit.py.analyze_social_patterns(...)`

**Output**: 
The resulting files are in `data/social_variation_analysis/reddit/`.
Each file has the following format:

```
[subreddit][tab][count conventional][tab][count non conventional]
```

Analyze them using `social_variation.rmd`


## Cosine similarity

Compute the cosine similarity between different pair types.

**Write the cosine similarities between each pair to a file**

Run the following:

```
python similarity_analysis.py
```

This prints out the similarities for the spelling variants, as well as BATS pairs and random pairs. The files are also provided in `data/similarity_analysis/`

**Statistical analysis**

See the R markdown file `cosine_similarity.rmd`

**Additional info about the pairs used for the baselines**

*Random pairs:*

This creates random pairs matched by frequency with the words in the spelling variant lists.

```
similarity_analysis.create_random_pairs(config.SPELLING_VARIANTS_FILES)
```

Output in: `spelling_variants/similarity_analysis/random_candidates.txt`

*BATS baseline*

The paper also compares against 2 pair types in the BATS dataset (https://vecto.space/projects/BATS/)

```
input_files_bats = [
         config.MAIN_DIR + "data/BATS_3.0/1_Inflectional_morphology/I01 [noun - plural_reg].txt",
         config.MAIN_DIR + "data/BATS_3.0/1_Inflectional_morphology/I06 [verb_inf - Ving].txt"
    ]
```


## Diagnostic classifiers

The output files are provided in `data/auxiliary_tasks`.
Each folder contains a file `classification_results.txt` with the results.

**Create dataset**
Create a dataset with word pairs and labels.
Run:

```
python auxiliary_tasks.py create_data
```

This creates a file called `classification_dataset.txt` in the format;

```
[unconventional form][tab][label (int)][conventional form]
```

The file is also provided in: `data/auxiliary_tasks/classification_dataset.txt`

Label mapping:

- 0: British vs. American
- 1: g-dropping
- 2: swapped
- 3: vowel omission
- 4: keyboard substitution
- 5: lengthening
- 6: common misspellings


**Run the experiments**

* **Baseline 1** (only the vectors for the conventional form)
	* Create features vectors: `python auxiliary_tasks.py create_data_control1`
	* Run experiments: `python auxiliary_tasks.py run_control1`
* **Baseline 2** (only the vectors for the conventional form + freq)
	* Create features vectors: `python auxiliary_tasks.py create_data_control2`
	* Run experiments: `python auxiliary_tasks.py run_control2`
* **Main setting** (based on vector differences)
	* Create features vectors: `python auxiliary_tasks.py create_data_diff`
	* Run experiments: `python auxiliary_tasks.py run_diff`
* **Majority classifier**
	* Run: ` python auxiliary_tasks.py run_majority`






## Analogy experiments

**Data**

* Analogy pairs: `data/analogies`
* Analogy results: `data/analogies_results`

**Dataset creation**

The analogy data was created by pairing each pair
with 10 randomly selected pairs within each category.

```
python analogy_experiment.py create_data
```

**Run experiments**


* Run skipgram: `python analogy_experiment.py run_skipgram --results_dir=[outputdir]`
* Run fastText: `python analogy_experiment.py run_fasttext --results_dir=[outputdir]`

**Evaluate**

Print a summary table using:

```
python analogy_experiment.py evaluate --results_dir=[input_dir]
```

`input_dir` should contain the output files of the analogy experiments (for example,  `data/analogies_results`).
Writes the output to a file called `merged_analogy_results.csv`

Analyze this file using the R markdown `analogy_analysis.rmd`


## Files:


**Config**

* `config.py`: Paths to files and directories
* `environment.yml` created using `conda env export ..`
* `config.ini`: Place the path to the main directory here


**Training embeddings**

* `gensim_word2vec.py`: Train Word2vec embeddings using gensim
* `train-embeddings.sh`: Train FastText for Twtitter (simialr file for Reddit)

**Embeddings**

* `embeddings_util.py`: Utility functions to work with embeddings (e.g., normalize, find neighbors, etc.)
* `EmbeddingWrapper.py`: Wrapper to work with both FastText and Gensim embeddings
* `EmbeddingsTestCase.py` Testing embedding implementation.
* `embeddings_analogies_tests.py`: Small file for additioanl testing.

**Notebooks**

* `introduction_example.ipynb`: notebook to generate the plot in the introduction.

**Analogies**

* `analogy_experiment.py`: Code to run the analogy experiments.
* `analogy_analysis.rmd`: Generate plots for the paper

**Similarity analysis**

* `similarity_analysis.py`: Compute cosine similarities between standard and non-standard variants.
* `cosine_similarity.rmd`: R script to analyze the cosine similarities

**Diagnostic classifiers**

* `auxiliarity_tasks.py`: Contains code to run classification experiments ('diagnostic classifiers')

**Social variation**

* `social_variation.rmd` R markdown to analyze the output.
* `process_reddit.py`: Subreddit analysis

**Utilities**

* `util.py`
* `DataWrapper.py`

