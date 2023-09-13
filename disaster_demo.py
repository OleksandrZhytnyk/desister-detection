import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tensorflow import keras
import numpy as np


def sentence_preprocessing(model_enb, tokenizer, list_sentences):
    tokenized = tokenizer(list_sentences, padding='max_length', max_length=args.max_length)
    input_ids = torch.tensor(tokenized["input_ids"])
    attention_mask = torch.tensor(tokenized["attention_mask"])
    with torch.no_grad():
        last_hidden_states = model_enb(input_ids=input_ids, attention_mask=attention_mask)
    features = last_hidden_states[1][-1][:, 0, :].numpy()
    return features


def custom_model(model_name):
    if model_name == 'lstm':
        model = keras.models.load_model('./custom_models/lstmnn.ckpt', compile=False)
    if model_name == 'gru':
        model = keras.models.load_model('./custom_models/grunn.ckpt', compile=False)
    if model_name == 'nn':
        model = keras.models.load_model('./custom_models/densenn.ckpt', compile=False)
    return model


def main():
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model_enb = BertForSequenceClassification.from_pretrained(args.model_path, output_hidden_states=True)

    def disasters_analysis(text):
        features = sentence_preprocessing(model_enb, tokenizer, [text])
        model = custom_model(args.model_name)
        predictions = model.predict(features)
        output = np.where(predictions > args.threshold, 1, 0)
        if output == 1:
            return "This tweet contains information about disaster"
        else:
            return "This tweet doesn`t contains information about disaster"

    demo = gr.Interface(
        fn=disasters_analysis,
        inputs='text',
        outputs="label",
        interpretation="default",
        examples=[["This is wonderful!"]])

    demo.launch()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
    parser.add_argument('--model_path', type=str, help='A path to the HF model', required=True)
    parser.add_argument('--max_length', type=int, help='Max len for the sentence preprocessing')
    parser.add_argument('--threshold', type=int, default=0.6, help='A threshold for the prediction')

    #  python .\custom_models\custom_nn.py --save_dir ./custom_models --model_name densenn --csv_origin_path data/train.csv --csv_emb_path data/train_emb.csv
    args = parser.parse_args()

    main()
