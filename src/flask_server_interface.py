import requests, json


def get_similar_words(word, server="http://localhost:5000/similar"):
    header = {"content-type": "application/json"}
    content = json.dumps({"word": word})
    answer = requests.post(server, data=content, headers=header)
    return answer.text


if __name__ == "__main__":
    interactive = True
    while interactive:
        word = input("Enter word (x to exit): ")
        if word != "x":
            similar_words = json.loads(get_similar_words(word))
            if "ERROR" in similar_words:
                print(f"ERROR: {similar_words['ERROR']}")
            else:
                for word in similar_words:
                    print(f"{word[0]}: {word[1]}")
            print("\n")

        else:
            interactive = False
    print(f"Nice to see you! Hope you had fun!")

