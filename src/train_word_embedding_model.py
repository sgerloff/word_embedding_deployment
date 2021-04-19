import argparse, gensim, json
import tensorflow as tf

from src.data_preprocessing import get_processed_data_from_csv


def instance_from_string(module_path):
    path_list = module_path.split(".")
    class_name = path_list[-1]
    directory_of_module = ".".join(path_list[:-1])
    module = __import__(directory_of_module, fromlist=[class_name])
    return getattr(module, class_name)


def get_label_encoder_and_decoder(data, label_column="Sentiment"):
    label_encoder = {}
    label_decoder = {}
    for index, label in enumerate(data[label_column].unique()):
        label_encoder[label] = index
        label_decoder[index] = label

    return label_encoder, label_decoder


def split_datasets(data, ratio=0.8, feature_column="OriginalTweet", label_column="Label"):
    split_index = int(ratio * len(data))
    train_df, test_df = data[:split_index], data[split_index:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_df[feature_column], train_df[label_column]))
    test_ds = tf.data.Dataset.from_tensor_slices((test_df[feature_column], test_df[label_column]))

    return train_ds, test_ds


def get_word_vectors(model):
    vocabulary = model.get_layer("vectorize").get_vocabulary()
    embeddings = model.get_layer("embedding").get_weights()[0]
    word_vectors = gensim.models.KeyedVectors(embeddings.shape[1])

    word_vectors.add(vocabulary, embeddings)
    return word_vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NN for sentiment classification, to create embeddings.')
    parser.add_argument('--csv', type=str, default='data/sample_data.csv',
                        help='Path to csv containing original tweets.')
    parser.add_argument('--save', type=str, default="data/global_average",
                        help='Base name for saved model and word vectors.')
    parser.add_argument('--factory', type=str,
                        default="src.word_embedding_model_factory.GlobalPoolingWordEmbeddingModelFactory",
                        help="Module of model factory")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train")
    parser.add_argument('--final', type=bool, default=False, help="Toggle for final run -> No test train split.")

    args = parser.parse_args()

    processed_data = get_processed_data_from_csv(args.csv)
    label_encoder, label_decoder = get_label_encoder_and_decoder(processed_data, label_column="Sentiment")
    processed_data["Label"] = processed_data["Sentiment"].apply(lambda x: label_encoder[x])

    # Shuffle data:
    processed_data = processed_data.sample(frac=1.)


    #Define training data and callbacks:
    if args.final:
        train_ds = tf.data.Dataset.from_tensor_slices((processed_data["OriginalTweet"], processed_data["Label"]))
        train_ds = train_ds.batch(64).shuffle(10000)
        test_ds=None
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                "/tmp/checkpoint",
                monitor="loss",
                save_weights_only=True,
                save_best_only=True
            )
        ]
    else:
        train_ds, test_ds = split_datasets(processed_data, ratio=0.8)
        train_ds = train_ds.batch(64).shuffle(10000)
        test_ds = test_ds.batch(64).shuffle(10000)
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                "/tmp/checkpoint",
                monitor="val_accuracy",
                mode="max",
                save_weights_only=True,
                save_best_only=True
            )
        ]


    #Build model from factory:
    factory = instance_from_string(args.factory)(
        vocabulary_size=40000,
        embedding_dim=64,
        sequence_length=300
    )
    model = factory.get_model(len(label_encoder), dense_sizes=[64])


    model.get_layer("vectorize").adapt(train_ds.map(lambda x, y: x))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics="accuracy"
    )

    history = model.fit(train_ds,
                        validation_data=test_ds,
                        epochs=args.epochs,
                        callbacks=callbacks)

    #Reinstatiate best model and save for server:
    model.load_weights("/tmp/checkpoint")
    tf.keras.models.save_model(model, args.save+".model")

    #Create word vectors using gensims KeyedVectors class:
    word_vectors = get_word_vectors(model)
    word_vectors.save(args.save + ".wv")

    #Save the label decoder for server:
    with open(args.save+".label_decoder", "w") as file:
        json.dump(label_decoder, file)
