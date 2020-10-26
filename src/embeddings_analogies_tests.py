
import EmbeddingWrapper

# Some additional tests to compare outputs
embedding_model = EmbeddingWrapper.EmbeddingWrapper("fasttext", "reddit", 300)

standard1, standard2, nonstandard1, nonstandard2 = ("great", "stadium", "grest", "stadiun")
#("those", "exactly", "thoes",  "excatly")
#("real",	"never", "reallll", "neverrrr")

# fallin - falling = startin - starting
# falling = fallin - startin + starting

print("\nWith no normalization")
# pos1, pos2, neg1
print(embedding_model.get_analogy(standard2, nonstandard1, 
                                nonstandard2, 10, standard1, normalize_before_query=False))

print("\nWith normalization")
print(embedding_model.get_analogy(standard2, nonstandard1, 
                                nonstandard2, 10, standard1, normalize_before_query=True))

print("\nFasttext implementation")
# model.get_analogies("berlin", "germany", "france")
print(embedding_model.f.get_analogies(standard2, nonstandard2, nonstandard1))



