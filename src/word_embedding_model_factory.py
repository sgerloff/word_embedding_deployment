import tensorflow as tf


class GlobalPoolingWordEmbeddingModelFactory:
    def __init__(self, vocabulary_size=40000, embedding_dim=64, sequence_length=300):
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length

        self.embedding_layer = tf.keras.layers.Embedding(self.vocabulary_size,
                                                         self.embedding_dim,
                                                         name="embedding")

        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            standardize=None,
            max_tokens=self.vocabulary_size,
            output_mode="int",
            output_sequence_length=self.sequence_length,
            name="vectorize"
        )

    def get_model(self, output_dim, dense_sizes=[64]):
        architecture = [
            self.vectorize_layer,
            self.embedding_layer,
            self.get_sequence_layers()
        ]

        for size in dense_sizes:
            architecture.append(
                tf.keras.layers.Dense(size, activation="relu")
            )

        architecture.append(
            tf.keras.layers.Dense(output_dim, activation="softmax")
        )

        return tf.keras.models.Sequential(architecture)

    def get_sequence_layers(self):
        return tf.keras.layers.GlobalAveragePooling1D()


class LSTMWordEmbeddingModelFactory(GlobalPoolingWordEmbeddingModelFactory):
    def get_sequence_layers(self):
        return tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.embedding_dim)
        )