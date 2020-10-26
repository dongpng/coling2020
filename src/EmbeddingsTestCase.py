import unittest
import numpy as np 
import numpy.testing as npt

import EmbeddingWrapper
import embeddings_util

class EmbeddingsTestCase(unittest.TestCase):

    def test_fasttext_nn(self):
        """ Make sure results match the command line tool of fasttext
        
        ./fasttext nn "fasttext_london_tweets_processed_may2018_april2019_embeddings_20_300_5.bin"

        ./fasttext analogies "fasttext_london_tweets_processed_may2018_april2019_embeddings_20_300_5.bin"

        """

        embedding_model = EmbeddingWrapper.EmbeddingWrapper("fasttext", "twitter", 300)
        
        results = embedding_model.get_most_similar("girl", 10)
        
        # Result from the command line tool
        # girlz 0.677679
        #gal 0.667554
        #boy 0.662205
        #girls 0.656006
        #girll 0.655023
        #woman 0.638082
        #bbygirl 0.606997
        #girlll 0.60681
        #3girllll 0.595928
        #girlfriend 0.592626

        top_words_correct5 = ["girlz", "gal", "boy", "girls", "girll"]
        top_scores_correct5 = [0.677679, 0.667554, 0.662205, 0.656006, 0.655023]

        for i, word in enumerate(top_words_correct5):
            npt.assert_equal(results[i][0], word)
            npt.assert_almost_equal(results[i][1], 
                                    top_scores_correct5[i], decimal=3)
        
        npt.assert_equal(len(results), 10)

        # Check single calculation
        single_result = np.dot(embedding_model.get_norm_embedding("girl"), 
                                embedding_model.get_norm_embedding(top_words_correct5[3]))
        npt.assert_almost_equal(single_result, results[3][1])


        # The code was developed with fasttext 0.8, 
        # before some of these functionalities were provided in Python
        results = embedding_model.get_most_similar("going", 10)
        results_fasttext_impl = embedding_model.f.get_nearest_neighbors('going')

        for i, (score, word) in enumerate(results_fasttext_impl):
            npt.assert_equal(results[i][0], 
                                    word)
            npt.assert_almost_equal(results[i][1], 
                                    score, decimal=3)  


        #Query triplet (A - B + C)? brother sister sis
        #bro 0.737392
        #brooo 0.629325
        #broooo 0.613188
        #fam 0.585725
        #guyyy 0.580572
        #brooooo 0.579555
        #brotha 0.569735
        #broooooo 0.568193
        #dawggg 0.561648
        #broski 0.558858

        result = embedding_model.get_analogy("brother", "sis", "sister", 
                                             5, normalize_before_query=False)
        npt.assert_equal(result[0][0], "bro")
        npt.assert_almost_equal(result[0][1], 0.737392, decimal=3)
        npt.assert_equal(result[2][0], "broooo")
        npt.assert_almost_equal(result[2][1], 0.613188, decimal=3)
        npt.assert_equal(len(result), 5)

        # Seems like analogies in cmd tool of fasttext (0.8) was without normalization,
        # but Python implementation is, matching gensim's behavior
        results_fasttext_impl = embedding_model.f.get_analogies("goin", "going", "doing")
        results = embedding_model.get_analogy("doing", "goin", "going", 
                                             10, normalize_before_query=True)

        for i, (score, word) in enumerate(results_fasttext_impl):
            npt.assert_equal(results[i][0], 
                                    word)
            npt.assert_almost_equal(results[i][1], 
                                    score, decimal=3)  

       

    def test_word2vec_similarity(self):
        w2vec_twitter = EmbeddingWrapper.EmbeddingWrapper("word2vec", "twitter", 300)
        gensim_result = w2vec_twitter.get_most_similar("brother", 10)

        # Own implementation
        _, top_scores = embeddings_util.find_nearest_neighbor(np.copy(w2vec_twitter.f.wv['brother']), 
                                                            w2vec_twitter.f.wv.vectors_norm, top_n=20)

        # topscores is reversed, skip first one because that's the query word
        npt.assert_almost_equal([b for (a,b) in gensim_result], top_scores[-11:-1][::-1]) 



    def test_word2vec_analogy(self):
        """ Test if analogy implementation is the same """
      
        w2vec_twitter = EmbeddingWrapper.EmbeddingWrapper("word2vec", "twitter", 300)
        
        gensim_result = w2vec_twitter.get_analogy("sis", "brother", "sister", 10)

        #check one result
        query = (w2vec_twitter.get_norm_embedding("brother") +  w2vec_twitter.get_norm_embedding("sis")
                 - w2vec_twitter.get_norm_embedding("sister"))
        query /= np.linalg.norm(query.astype(float))
        npt.assert_almost_equal(np.dot(
            query,
            w2vec_twitter.get_norm_embedding(gensim_result[3][0])
        ), gensim_result[3][1])


        # Own implementation
        own_result = (embeddings_util.find_analogy("brother", "sister", "sis", 
                                                     w2vec_twitter, 
                                                     w2vec_twitter.f.wv.vectors_norm, 
                                                     top_n=10, search_word=None, 
                                                     normalize_before_query=True))

        for id, pair in enumerate(own_result):
            word, score = pair
            self.assertEqual(word, gensim_result[id][0])
            npt.assert_almost_equal(score, gensim_result[id][1])


if __name__ == '__main__':
    unittest.main()