from flask import Flask, jsonify, request

import gensim

app = Flask(__name__)


@app.route('/similar', methods=["POST"])
def similar():
    print(request.json)
    word = request.json['word']
    return jsonify(word_vectors.most_similar(word))


if __name__ == "__main__":
    word_vectors = gensim.models.KeyedVectors.load("data/word2vec.wv")
    app.run(debug=True)
