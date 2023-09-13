from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch


def main():
    # Load pretrained model/tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertForSequenceClassification.from_pretrained(args.model_path, output_hidden_states=True)
    print("Model loaded")
    df = pd.read_csv(args.csv_path)
    df = df[['text', 'target']]
    tokenized = tokenizer(df["text"].tolist(), truncation=True, padding=True, max_length=args.max_length)
    print("Tokenizer Loaded")
    input_ids = torch.tensor(tokenized["input_ids"])
    attention_mask = torch.tensor(tokenized["attention_mask"])

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    print("Start saving features")

    features = last_hidden_states[1][-1][:,0,:].numpy()
    pd.DataFrame(features).to_csv(args.save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', type=str, help='A path to the folder for saving csv file', required=True)
    parser.add_argument('--model_path', type=str, help='A path to the checkpoints folder', required=True)
    parser.add_argument('--max_length', type=int, default=256, help='Max len for the sentence preprocessing')
    parser.add_argument('--csv_path', type=str, default='', help='A path to the csv file')
    args = parser.parse_args()

    main()