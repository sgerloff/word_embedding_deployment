from src.data_preprocessing import get_processed_data_from_csv
import argparse, codecs, gensim


def get_word2vec(path_to_sentences, **kwargs):
    sentences = gensim.models.word2vec.LineSentence(path_to_sentences)
    model = gensim.models.Word2Vec(
        sentences=sentences,
        **kwargs
    )
    return model


def save_sentences(path_to_csv, save="/tmp/sentences"):
    processed_data = get_processed_data_from_csv(path_to_csv)
    sentence_list = processed_data["OriginalTweet"].to_list()
    with codecs.open(save, "w", "utf-8-sig") as file:
        file.write("\n".join(sentence_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates word vectors from tweets.')
    parser.add_argument('--csv', type=str, default='../data/sample_data.csv',
                        help='Path to csv containing original tweets.')
    parser.add_argument('--save', type=str, default="../data/word2vec",
                        help='Base name for saved model and word vectors.')

    args = parser.parse_args()

    save_sentences(args.csv, save="/tmp/sentences")
    word2vec = get_word2vec("/tmp/sentences",
                            vector_size=100,
                            window=5
                            )

    word2vec.wv.save(args.save + ".wv")

    print(f"\nWords most similar to {'advice'}:\n")
    for word, similarity in word2vec.wv.most_similar('advice'):
        print(f"{word}: {similarity}")
