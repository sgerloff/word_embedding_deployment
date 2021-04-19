from flask import Flask, jsonify, request

import gensim, json, os
import tensorflow as tf
from src.data_preprocessing import Preprocessor
from src.spell_correction import SpellChecker


app = Flask(__name__)


def get_word_vector_dict(path):
    filenames = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    word_vector_files = [file for file in filenames if file.endswith(".wv")]
    return {os.path.splitext(file)[0]: gensim.models.KeyedVectors.load(os.path.join(path, file)) for file in
            word_vector_files}


def get_model_dict(path):
    dirnames = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    model_paths = [dir for dir in dirnames if dir.endswith(".model")]
    model_dict = {}
    for file in model_paths:
        basename = os.path.splitext(file)[0]
        decoder_file = os.path.join(path, basename + ".label_decoder")
        if os.path.isfile(decoder_file):
            model_dict[basename] = {
                "model": tf.keras.models.load_model(os.path.join(path, file)),
                "decoder": load_decoder(decoder_file)
            }
    return model_dict


def load_decoder(path):
    with open(path, "r") as file:
        decoder = json.load(file)
    return decoder


@app.route('/similar', methods=["POST"])
def similar():
    word = request.json['word']
    model = request.json['model']

    if model in word_vector_dict:
        word_vector = word_vector_dict[model]
        word = preprocessor(word)
        if word in word_vector.key_to_index:
            return jsonify(word_vector.most_similar(word))
        else:
            spell = SpellChecker(list(word_vector.key_to_index.keys()))
            most_similar = spell.get_most_similar_words(word, n=3)
            return jsonify({"ERROR": f"word '{word}' is not known! Did you mean: {most_similar}?"})
    else:
        return jsonify({"ERROR": f"model '{model}' is not known!"})


@app.route('/sentiment', methods=["POST"])
def sentiment():
    sentence = request.json["sentence"]
    model = request.json["model"]
    if model in model_dict:
        sentence = preprocessor(sentence)
        pred = model_dict[model]["model"].predict(
            tf.expand_dims(tf.constant(sentence), axis=0)
        )
        index = int(tf.math.argmax(pred[0]))
        return jsonify(model_dict[model]["decoder"][str(index)])
    else:
        return jsonify({"ERROR": f"{model} is not known!"})


if __name__ == "__main__":
    preprocessor = Preprocessor()

    word_vector_dict = get_word_vector_dict("data")
    model_dict = get_model_dict("data")
    app.run(host='0.0.0.0')
