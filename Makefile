
word2vec:
	python -m src.word2vec --csv="data/sample_data.csv" --save="data/word2vec"

global-average:
	python -m src.train_word_embedding_model --csv="data/sample_data.csv" --save="data/global_average" --factory="src.word_embedding_model_factory.GlobalPoolingWordEmbeddingModelFactory" --epochs=15 --final="True"

lstm:
	python -m src.train_word_embedding_model --csv="data/sample_data.csv" --save="data/lstm" --factory="src.word_embedding_model_factory.LSTMWordEmbeddingModelFactory" --epochs=5 --final="True"

setup-server:
	docker build -t word_embedding:v0.1 .

run-server:
	docker run -d -p 5000:5000 word_embedding:v0.1
