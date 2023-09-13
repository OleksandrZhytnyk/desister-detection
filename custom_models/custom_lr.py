import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle


def main():
    origin_data = pd.read_csv(args.csv_origin_path)
    embeddings = pd.read_csv(args.csv_emb_path)
    embeddings.drop(["Unnamed: 0"], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, origin_data["target"], test_size=0.25, random_state=42, stratify=origin_data["target"])

    model = LogisticRegression(penalty="l1", C=0.001, solver="liblinear", max_iter=50, random_state=42)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, prediction))

    print("Saving the model")

    pickle.dump(model, open(f'{args.save_dir}/{args.model_name}.pickle', "wb"))


# python .\custom_models\custom_lr.py --save_dir ./custom_models --model_name logreg --csv_origin_path data/train.csv --csv_emb_path data/train_emb.csv
# Accuracy:  0.8902310924369747


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, help='A path to the folder for saving checkpoints', required=True)
    parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
    parser.add_argument('--csv_origin_path', type=str, default='', help='A path to the origin csv file')
    parser.add_argument('--csv_emb_path', type=str, default='', help='A path to the embeddings csv file')

    args = parser.parse_args()
    main()
