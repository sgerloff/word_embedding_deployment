import requests


def get_similar_words(word, server="http://localhost:5000/similar"):
    header = {"content-type": "application/json"}
    content = {"word": word}
    answer = requests.post(server, data=content, headers=header)
    return answer


if __name__ == "__main__":
    interactive = True
    while interactive:
        word = input("Enter word (x to exit): ")
        if word != "x":
            similar_words = get_similar_words(word)
            for word in similar_words:
                print(f"{word[0]}: {word[1]}")
        else:
            interactive = False
    print(f"Nice to see you! Hope you had fun!")

