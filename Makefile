
word2vec:
	python -m src.word2vec --csv="data/sample_data.csv" --save="data/word2vec" --sentences="data/processed_sentences.txt"

setup-server:
	docker build -t word_embedding:v0.1 .

run-server:
	docker run -d -p 5000:5000 word_embedding:v0.1
