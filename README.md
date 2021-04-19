# Exploration: Deploy Word Embeddings

Motivated by a code challenge, this projects serves as a playground to explore word embeddings and their deployment.
The main objective is to train word embeddings from a corpus of tweets, which come labeled with respect to their sentiment.

The goal is to run a dockerized web server that can be used to query word embeddings.

## Word Embeddings

We explored two approaches to create word embedding: 
 1. Word2Vec
 2. Sentiment Classification

### Word2Vec

The first approach is Word2Vec, which is unsupervised.
Here, we employed the gensim implementation of the Word2Vec model training on a CBOW, i.e. continuous bag of words.
That is, the model is tasked to predict a word from its surrounding words.

To train the model on the available data simply execute:

```bash
make word2vec
```

This command will process the corpus of text, train a Word2Vec model for 5 epochs and write the embeddings such that they are available for the web server.

The strength of this approach is the straight-forward implementation and training.
However, this comes with two major drawbacks.
First, to evaluate the quality of the embeddings one would need to construct tests, such as [GLUE](https://gluebenchmark.com).
Second, the properties captured by the word embeddings are only with respect to the general semantic/usage of the words.
These two drawbacks, can be addressed by our second approach.

### Sentiment Classification

Often times word embeddings are used to aid another task, such as classification of texts.
Given labeled data, solving this task can be used to generate word embeddings directly relating the words to the final classification.
That is, instead of capturing which words are semantically similar, these word embeddings should capture how indicative the word is to a certain class.

Here, we are given a dataset from twitter which comes prelabeld with five different sentiments ranging from extremly negative to extremly positive.
Extracting the vocabulary from the collection of tweets, we can train a model to predict the sentiment of these tweets, by learning word embeddings and the weights of a classifier. 
As a result, the learned word embeddings are expected to have properties more useful for the sentiment classification of tweets.

To this end we build a very small model, employing the tensorflow 'Embedding' layer, which is passed to a classifier.
For the classifier, we consider both a simple GlobalAveragePooling1D layer as well as a bidrirectional LSTM.
In both cases the top of the model consists of a Dense layer and the final softmax layer.
We observe that the LSTM performs slightly better on the classification task, achiving a final accuracy:

| Model | Test Accuracy |
| --- | --- |
| GlobalAveragePooling1D| 70.9% |
| Bidirectional LSTM | 74.5% |

To train the final models simply execute:

```bash
make global-average
make lstm
```

## Deployment

Having trained word embeddings you want users or other applications to interact with them.
One way to achieve this is to run a web server, which can be queried to return, for example, the most similar words to a user provided word, or the sentiment to a user provided text.

To run the web server it is best practice to wrap the application in a Docker container, fixing all dependencies of the operating system.
Here, the docker container is build from a Dockerfile, which inherits from the official tensorflow  Docker image and sets up the environment for a simple flask web server.

To build the Docker image, execute:

```bash
make setup-server
```

Afterwards, to run the server simply execute:

```bash
make run-server
```

### Usage

To interact with the web-server you will need to create requests.
For getting the 10 most similar words, such a request may look like this:

```bash
curl -i -H "Content-Type: application/json" -X POST -d '{"word": "good", "model": "word2vec"}' http://localhost:5000/similar
```

Whereas, for sentiment classification, you would need to post:

```bash
curl -i -H "Content-Type: application/json" -X POST -d '{"sentence": "This web-server works like a charm! This is amazing! #loveit @You look at me! http://t.co.url/asdf", "model": "lstm"}' http://localhost:5000/sentiment
```

However, this is cumbersome. To help you with exploring the web-server that we have implemented feel free to execute a little interactive helper script:

```bash
python -m src.flask_server_interface --data="data"
```

### Spell Correction

If you misspell a word the web-server performs a spell check and returns the most similar words that are in the vocabulary for the word vectors. 
Try them out and use them in your next query.