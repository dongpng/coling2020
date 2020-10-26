import gensim, logging, time

import config
import DataWrapper

DATASET = "Twitter"

MIN_COUNT = 20
WINDOW = 5

# Set up logging
if DATASET == "Twitter":
	logging.basicConfig(filename=config.TWITTER_EMBEDDINGS_DIR + 'gensim_word2vec_' + str(time.time()) + '.log',level=logging.INFO)
elif DATASET == "Reddit":
	logging.basicConfig(filename=config.REDDIT_EMBEDDINGS_DIR + 'gensim_word2vec_' + str(time.time()) + '.log',level=logging.INFO)
	
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

for dimension in [50,100,150,200,250,300]: 
	sentences = None
	if DATASET == "Twitter":
		sentences = DataWrapper.DataWrapper(config.TWITTER_PROCESSED_DATA) 
	elif DATASET == "Reddit":
		sentences = DataWrapper.DataWrapperGzip(config.REDDIT_PROCESSED_DATA)

	# use skipgram and negative sampling
	# sg: 1 for skip-gram; otherwise CBOW.
	# hs If 1, hierarchical softmax; 0 negative sampling 
	model = gensim.models.Word2Vec(sentences, size=dimension, min_count=MIN_COUNT, window=WINDOW, sg=1, hs=0, iter=5)
	
	if DATASET == "Twitter":
		model.save(config.TWITTER_EMBEDDINGS_DIR + 'word2vec_london_tweets_processed_may2018_april2019_embeddings_' + str(MIN_COUNT) + "_"+ str(dimension) + "_" + str(WINDOW))
	elif DATASET == "Reddit":
		model.save(config.REDDIT_EMBEDDINGS_DIR + 'word2vec_reddit_processed_2018_05_10_embeddings_' + str(MIN_COUNT) + "_"+ str(dimension) + "_" + str(WINDOW))

