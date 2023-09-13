from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from dataloader.NewsTweetDataset import NewsTweetDataset
from utils.utils import set_seed, compute_metrics, CustomCallback


def main():
    # set seed
    set_seed(1)

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name, do_lower_case=True)

    df = pd.read_csv(args.csv_path)

    df = df[['text', 'target']]

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(df['text'], df['target'], test_size=0.1,
                                                                            stratify=df['target'])

    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=args.max_length)
    valid_encodings = tokenizer(valid_texts.tolist(), truncation=True, padding=True, max_length=args.max_length)

    # convert our tokenized data into a torch Dataset
    train_dataset = NewsTweetDataset(train_encodings, train_labels.values)
    valid_dataset = NewsTweetDataset(valid_encodings, valid_labels.values)

    model = BertForSequenceClassification.from_pretrained(args.model_name).to(args.device)

    # define all arguments
    training_args = TrainingArguments(
        output_dir='./model',
        num_train_epochs=args.epochs,  # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        weight_decay=args.weight_decay,  # strength of weight decay
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        evaluation_strategy="steps",  # evaluate each `logging_steps`
    )

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # the callback that computes metrics of interest
    )
    trainer.add_callback(CustomCallback(trainer))
    trainer.train()

    model_path = args.save_dir
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, help='A path to the folder for saving checkpoints', required=True)
    parser.add_argument('--model_name', type=str, help='Name of the HuggingFace model', required=True)
    parser.add_argument('--device', type=str, help='Name of the HuggingFace model', required=True)
    parser.add_argument('--max_length', type=int, default=256, help='Max len for the sentence preprocessing')
    parser.add_argument('--csv_path', type=str, default='', help='A path to the csv file')
    parser.add_argument('--epochs', type=int, default=3, help='A number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size for dataloader')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='A weight decay for optimizer')

    args = parser.parse_args()
#  python main.py --save_dir ./checkpoint --model_name bert-base-cased --device cuda --max_length 384 --csv_path data/train.csv
    main()
