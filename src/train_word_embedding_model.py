import tensorflow as tf

from src.data_preprocessing import get_processed_data_from_csv
from src.word_embedding_model_factory import GlobalPoolingWordEmbeddingModelFactory


def get_label_encoder_and_decoder(data, label_column="Sentiment"):
    label_encoder = {}
    label_decoder = {}
    for index, label in enumerate(data[label_column].unique()):
        label_encoder[label] = index
        label_decoder[index] = label

    return label_encoder, label_decoder


def split_datasets(data, ratio=0.8, feature_column="OriginalTweet", label_column="Label"):
    split_index = int(ratio*len(data))
    train_df, test_df = data[:split_index], data[split_index:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_df[feature_column], train_df[label_column]))
    test_ds = tf.data.Dataset.from_tensor_slices((test_df[feature_column], test_df[label_column]))

    return train_ds, test_ds



if __name__ == "__main__":
    processed_data = get_processed_data_from_csv("../data/sample_data.csv")
    label_encoder, label_decoder = get_label_encoder_and_decoder(processed_data, label_column="Sentiment")
    processed_data["Label"] = processed_data["Sentiment"].apply(lambda x: label_encoder[x])

    #Shuffle data:
    processed_data = processed_data.sample(frac=1.)

    train_ds, test_ds = split_datasets(processed_data, ratio=0.8)
    train_ds = train_ds.batch(64).shuffle(10000)
    test_ds = test_ds.batch(64).shuffle(10000)

    factory = GlobalPoolingWordEmbeddingModelFactory(
        vocabulary_size=40000,
        embedding_dim=64,
        sequence_length=300
    )

    model = factory.get_model(len(label_encoder), dense_sizes=[64])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics="accuracy"
    )

    history = model.fit(train_ds, validation_data=test_ds, epochs=10)