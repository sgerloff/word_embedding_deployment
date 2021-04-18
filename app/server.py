from flask import Flask, jsonify, request

import gensim

app = Flask(__name__)


@app.route('/similar', methods=["POST"])
def similar():
    print(request.json)
    word = request.json['word']
    if word in word_vectors.key_to_index:
        return jsonify(word_vectors.most_similar(word))
    else:
        return jsonify({"ERROR": f"{word} is not known!"})


if __name__ == "__main__":
    word_vectors = gensim.models.KeyedVectors.load("word2vec.wv")
    app.run(host='0.0.0.0')
