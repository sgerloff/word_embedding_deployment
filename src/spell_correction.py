import pandas as pd


def get_ngrams(word, ngrams=2, padding=True):
    if padding:
        pads = ngrams - 1
        word = " " * pads + word + " " * pads
    ngram_list = []
    for i in range(len(word) - ngrams + 1):
        ngram_list.append(tuple(word[i:i + ngrams]))
    return set(ngram_list)


def jaccard_distance(set1, set2):
    union = len(set1.union(set2))
    intersect = len(set1.intersection(set2))
    return (union - intersect) / union


def get_word_distance(word1, word2, ngrams=2, padding=True):
    return jaccard_distance(
        get_ngrams(word1, ngrams=ngrams, padding=padding),
        get_ngrams(word2, ngrams=ngrams, padding=padding)
    )


class SpellChecker:
    def __init__(self, vocabulary, ngrams=2, padding=True):
        self.vocabulary = vocabulary
        self.ngrams = ngrams
        self.padding = padding
        self.precompute_vocabulary()

    def precompute_vocabulary(self):
        self.vocabulary = pd.DataFrame(self.vocabulary, columns=["word"])
        self.vocabulary["ngrams"] = self.vocabulary["word"].apply(
            lambda x: get_ngrams(x, ngrams=self.ngrams, padding=self.padding)
        )

    def get_most_similar_words(self, word, n=1):
        word_ngrams = get_ngrams(word, ngrams=self.ngrams, padding=self.padding)
        self.vocabulary["distance"] = self.vocabulary["ngrams"].apply(
            lambda x: jaccard_distance(x,word_ngrams)
        )
        return self.vocabulary.sort_values(by="distance", ascending=True)["word"].head(n).values
