import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


def create_model_LSTM():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32, input_shape=(768, 1)))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC(), 'accuracy'])
    return model

def main():
    origin_data = pd.read_csv(args.csv_origin_path)
    embeddings = pd.read_csv(args.csv_emb_path)
    embeddings.drop(["Unnamed: 0"], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, origin_data["target"], test_size=0.25, random_state=42, stratify=origin_data["target"])

    my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=10),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=f"{args.save_dir}/{args.model_name}.ckpt",
                        monitor='val_accuracy',
                        mode="max",
                        save_best_only=True)
                    ]

    model = create_model_LSTM()

    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_test, y_test),
                        batch_size=32,
                        callbacks=my_callbacks)

    plt.plot(history.history['val_auc'], label='valid')
    plt.plot(history.history['auc'], label='train')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='train')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='valid')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, help='A path to the folder for saving checkpoints', required=True)
    parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
    parser.add_argument('--csv_origin_path', type=str, default='', help='A path to the origin csv file')
    parser.add_argument('--csv_emb_path', type=str, default='', help='A path to the embeddings csv file')
    # Accuracy 0.9154
    #  python .\custom_models\custom_nn.py --save_dir ./custom_models --model_name densenn --csv_origin_path data/train.csv --csv_emb_path data/train_emb.csv
    args = parser.parse_args()
    main()
