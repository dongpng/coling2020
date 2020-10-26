import numpy as np 
import fasttext
import gensim

import config
import embeddings_util

MODEL_WORD2VEC = "word2vec"
MODEL_FASTTEXT = "fasttext"

### Wrapper for gensim and fasttext libraries ""
def get_embeddings(model, dataset, num_dim):
    """ Reads in the embeddings model
        model: fasttext or word2vec
        dataset: reddit or twitter
        num_dim: number of dimensions
    """
    f = None
    num_dim = str(num_dim)
    if model == "fasttext" and dataset=="reddit":
        f = fasttext.load_model(config.REDDIT_EMBEDDINGS_DIR + 
                                    "fasttext_reddit_processed_2018_05_10_cleaned_embeddings_20_" + num_dim + "_5.bin")
    elif model == "fasttext" and dataset=="twitter":
        f = fasttext.load_model(config.TWITTER_EMBEDDINGS_DIR + 
                                    "fasttext_london_tweets_processed_may2018_april2019_embeddings_20_" + num_dim + "_5.bin")
    elif model == "word2vec" and dataset=="twitter":
        f = gensim.models.Word2Vec.load(config.TWITTER_EMBEDDINGS_DIR + 
                                    'word2vec_london_tweets_processed_may2018_april2019_embeddings_20_' + num_dim + '_5')
    elif model == "word2vec" and dataset=="reddit":
        f = gensim.models.Word2Vec.load(config.REDDIT_EMBEDDINGS_DIR + 
                                    'word2vec_reddit_processed_2018_05_10_embeddings_20_' + num_dim + '_5')      
    else:
        print("Error, unknown model")
        print(model)
        print(dataset)
        print(num_dim)

    return f


class EmbeddingWrapper:
    def __init__(self, model, dataset, num_dim):
        """ model: MODEL_WORD2VEC, MODEL_FASTTEXT
        dataset: reddit or twitter
        num_dim: Number of dimensions """
        self.model = model
        self.dataset = dataset
        self.num_dim = num_dim
        self.f = get_embeddings(model, dataset, num_dim)
        if model == MODEL_FASTTEXT:
            self.f_vectors = embeddings_util.get_normalized_vectors(self.f)

    def get_norm_embedding(self, word):
        """ Return normalized embedding for given word """
        a_norm = self.get_embedding(word)
        a_norm /= np.linalg.norm(a_norm.astype(float))
        return a_norm

    def get_embedding(self, word):
        """ Return the embedding for a given word """
        if self.model == MODEL_WORD2VEC:
            return np.array(self.f.wv[word])
        elif self.model == MODEL_FASTTEXT:
            return self.f.get_word_vector(word)

    def get_vocab(self):
        """return list of words """
        if self.model == MODEL_WORD2VEC:
            return list(self.f.wv.vocab)
        elif self.model == MODEL_FASTTEXT:
            return self.f.get_words()

    def index2word(self, index):
        """return list of words """
        if self.model == MODEL_WORD2VEC:
            return self.f.wv.index2word[index]
        elif self.model == MODEL_FASTTEXT:
            return self.f.get_words()[index]

    def word2index(self, word):
        if self.model == MODEL_WORD2VEC:
            return self.f.wv.vocab.get(word).index
        elif self.model == MODEL_FASTTEXT:
            return self.f.get_word_id(word)


    def get_analogy(self, pos1, pos2, neg1, num_words, search_word=None, normalize_before_query=True):
        """Return analogy results
        pos1 - neg1 + pos2 = ?
        num_words: max. number of words to return. Input is excluded
        search_word: if not None, only return result up to search_word (more efficient?)
        return list with [(word, score)]
        """
        if self.model == MODEL_WORD2VEC:
            return self.f.wv.most_similar(positive=[pos1, pos2], negative=[neg1], topn=num_words)
        elif self.model == MODEL_FASTTEXT:
            ## command line tool implementation equals no normalization before query.
            # Here True for consistency with gensim.
            return embeddings_util.find_analogy(pos1, neg1, pos2, self, self.f_vectors, 
                                                top_n=num_words, search_word=search_word, 
                                                normalize_before_query=normalize_before_query)


    def get_most_similar(self, word, num_words):
        if self.model == MODEL_WORD2VEC:
            return self.f.wv.most_similar(positive=[word], topn=num_words)
        elif self.model == MODEL_FASTTEXT:
            return embeddings_util.get_top_words(word, self.f, self.f_vectors, top_n=num_words)
