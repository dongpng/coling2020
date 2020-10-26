import numpy as np
from sklearn.decomposition import PCA

#### Utility functions for working with embeddings """"

def get_normalized_vectors(f):
    """ modified from https://github.com/facebookresearch/fastText/blob/master/python/doc/examples/compute_accuracy.py
    f: the fasttext model
    """

    words, _ = f.get_words(include_freq=True)
        
    vectors = np.zeros((len(words), f.get_dimension()), dtype=float)
    for i in range(len(words)):
        wv = f.get_word_vector(words[i])
        wv = wv / float(np.linalg.norm(wv.astype(float))) 
        vectors[i] = wv

    return vectors



def find_nearest_neighbor(query, vectors, cossims=None, top_n=50):
    """
    #heavily modified from https://github.com/facebookresearch/fastText/blob/master/python/fastText/util/util.py
    query is a 1d numpy array corresponding to the vector to which you want to
    find the closest vector
    vectors is a 2d numpy array corresponding to the vectors you want to consider
    cossims is a 1d numpy array of size len(vectors), which can be passed for efficiency
    returns the indices of the closest matches and their similarity scores
    """
    queryNorm = np.linalg.norm(query.astype(float))
    query /= queryNorm
    
    if cossims is None:
        cossims = np.matmul(vectors, query, out=cossims)
    else:
        np.matmul(vectors, query, out=cossims)
    
    sorted_indices = np.argsort(cossims)
    if top_n:
        return sorted_indices[-top_n:], cossims[sorted_indices[-top_n:]]
    else:
        return sorted_indices, cossims[sorted_indices]
        
    
def get_top_words_query(query, model, vectors, ignore_set=set(), top_n=50, to_print=False):
    """ Return top words and top scores
    query: the query vector (1d numpy array)
    model: the fasttext model
    vectors: vectors to consider
    ignore_set: are there words we want to exclude?
    top_n: top n to consider
    to_print: if true, top words and their scores will be printed.
    
     """
     # need to add len(ignore_set) to make sure enough results are returned in case ignore is in the top n
    nearest_words_ids, nearest_words_scores = find_nearest_neighbor(query, vectors, top_n=top_n + len(ignore_set))
    all_words = model.get_words()

    result = []
    top_words = []
    top_scores = []
    for idx, word_id in enumerate(nearest_words_ids[::-1]): #reverse order
        if all_words[word_id] in ignore_set:
            continue
        result.append((all_words[word_id],
                       nearest_words_scores[-idx-1]))
        if to_print:
            print("%s\t%s" % (top_words[-1], top_scores[-1]))

        if len(result) == top_n:
            break
    return result



def get_top_words(word, model, vectors, to_print=False, top_n=50):
    """ Given a word, return the top most similar words and their scores
        and print the results 
    """
    query = model.get_word_vector(word)
    return get_top_words_query(query, model, vectors, ignore_set=set(list([word])), 
                               top_n=top_n, to_print=to_print)


def cosine_similarity(a,b, f):
    """ Cosine similarity between a and b"""
    a_norm = f.get_word_vector(a)
    a_norm /= np.linalg.norm(a_norm.astype(float))

    b_norm = f.get_word_vector(b)
    b_norm /= np.linalg.norm(b_norm.astype(float))

    return a_norm.dot(b_norm)

def find_analogy(a,b,c, wrapper, vectors, top_n=None, 
                 search_word=None, normalize_before_query=True):
    """a - b + c = ?
    embedding_wrapper: Embedding wrapper
    vector: fasttext vectors 
    """
    query = [a,b,c]
    if normalize_before_query:
        query = [wrapper.get_norm_embedding(x) for x in query]
    else:
        query = [wrapper.get_embedding(x) for x in query]
    #query = [f.get_word_vector(x) for x in query]
    query = query[0] - query[1] + query[2]
    
    # normalize (not necessarily needed, already done by find_nearest_neighbor)
    query /= np.linalg.norm(query.astype(float))

    # we want to ignore words in the query                
    ignore = set()
    ignore.add(wrapper.word2index(a))
    ignore.add(wrapper.word2index(b))
    ignore.add(wrapper.word2index(c))
    
    # find the nearest neighbors
    # add len(ignore) to make sure enough results are returned top return top n
    nearest_words_ids, scores = find_nearest_neighbor(query, vectors, top_n=top_n + len(ignore))
    
    results = []
    for idx, word_id in enumerate(nearest_words_ids[::-1]):
        if word_id in ignore:
            continue
        
        word = wrapper.index2word(word_id)
        
        results.append((word, scores[-idx-1]))
        if search_word is not None and word == search_word:
            break

        if len(results) == top_n:
            break
        
    return results