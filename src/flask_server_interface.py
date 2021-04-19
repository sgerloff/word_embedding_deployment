import requests, json, argparse

from src.server import get_model_dict, get_word_vector_dict

def get_similar_words(word, model="word2vec", server="http://localhost:5000/similar"):
    header = {"content-type": "application/json"}
    content = json.dumps({"word": word, "model": model})
    answer = requests.post(server, data=content, headers=header)
    return answer.text

def similar_words_loop(data):
    model_choices = get_word_vector_dict(data).keys()
    model = input(f"Choose model {model_choices}: ")
    interactive = True
    while interactive:
        word = input("Enter word (x to exit): ")
        if word != "x":
            similar_words = json.loads(get_similar_words(word, model=model))
            if "ERROR" in similar_words:
                print(f"ERROR: {similar_words['ERROR']}")
            else:
                for word in similar_words:
                    print(f"{word[0]}: {word[1]}")
            print("\n")

        else:
            interactive = False
    print(f"Nice to see you! Hope you had fun!")


def get_sentiment(sentence, model="global_average", server="http://localhost:5000/sentiment"):
    header = {"content-type": "application/json"}
    content = json.dumps({"sentence": sentence, "model": model})
    answer = requests.post(server, data=content, headers=header)
    return answer.text


def sentiment_loop(data):
    model_choices = get_model_dict(data).keys()
    model = input(f"Choose model {model_choices}: ")
    interactive = True
    while interactive:
        sentence = input("Enter tweet (x for exit): ")
        if sentence != "x":
            sentiment = json.loads(get_sentiment(sentence, model=model))
            print(f"Sentiment: {sentiment}")
        else:
            interactive = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple commandline interface to interact with flask server.')
    parser.add_argument('--data', type=str, default='../data',
                        help='path to model and word vector files')
    args = parser.parse_args()

    interactive = True

    query_type = input("What do you want to query [similar, sentiment]? ")
    if query_type == "similar":
        similar_words_loop(args.data)
    if query_type == "sentiment":
        sentiment_loop(args.data)



