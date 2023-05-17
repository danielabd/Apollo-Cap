import os
import pickle

import numpy as np
from torch import nn
from transformers import Trainer, TrainingArguments
# from transformers import TextClassificationDataset
from transformers import TextClassificationPipeline
from datasets import Dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer # SENTIMENT

from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from huggingface_hub import HfFolder, notebook_login


def use_source_tutorial():
    model_id = "roberta-base"
    dataset_id = "ag_news"
    # replace the value with your model: ex <hugging-face-user>/<model-name>
    repository_id = "achimoraites/roberta-base_ag_news"

    # Load dataset
    dataset = load_dataset(dataset_id)

    # Training and testing datasets
    train_dataset = dataset['train']
    test_dataset = dataset["test"].shard(num_shards=2, index=0)

    # Validation dataset
    val_dataset = dataset['test'].shard(num_shards=2, index=1)

    # Preprocessing
    tokenizer = RobertaTokenizerFast.from_pretrained(model_id, use_auth_token=access_token)

    # This function tokenizes the input text using the RoBERTa tokenizer.
    # It applies padding and truncation to ensure that all sequences have the same length (256 tokens).
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True, max_length=256)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    # Set dataset format
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # We will need this to directly output the class names when using the pipeline without mapping the labels later.
    # Extract the number of classes and their names
    num_labels = dataset['train'].features['label'].num_classes
    class_names = dataset["train"].features["label"].names
    print(f"number of labels: {num_labels}")
    print(f"the labels: {class_names}")

    # Create an id2label mapping
    id2label = {i: label for i, label in enumerate(class_names)}

    # Update the model's configuration with the id2label mapping
    config = AutoConfig.from_pretrained(model_id)
    config.update({"id2label": id2label})

    model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)

    return model, train_dataset, val_dataset


def replace_user_home_dir(path):
    if str(path)[0] == '~':
        path = os.path.join(os.path.expanduser('~'), path[2:])
    elif str(path).split('/')[1] == 'Users':
        path = os.path.join(os.path.expanduser('~'), "/".join(path.split('/')[3:]))
    elif '/' in str(path) and str(path).split('/')[1] == 'home':
        if str(path).split('/')[2] == 'bdaniela':
            path = os.path.join(os.path.expanduser('~'), "/".join(path.split('/')[3:]))
        else:
            path = os.path.join(os.path.expanduser('~'), "/".join(path.split('/')[4:]))
    return path


def get_annotations_data(annotations_path, data_split):
    '''
    :param annotations_path: dictionary:keys=dataset names, values=path to pickle file
    factual_captions: need to b none for flickrstyle10k
    :return: gts_per_data_set: key=img_name,values=dict:keys=['image_path','factual','humor','romantic','positive','negative'], values=gt text
    '''
    texts_list = []
    labels_list = []
    with open(os.path.join(annotations_path,data_split+'.pkl'), 'rb') as r:
        data = pickle.load(r)
    for k in data:
        # if len(labels_list)>10: #todo: remove
        #     break #todo: remove
        for style in data[k]:
            if style != 'factual' and style != 'image_path':
                for caption in data[k][style]:
                    texts_list.append(caption)
                    labels_list.append(style)
    return texts_list, labels_list


def main():
    ##############
    if True:
        id2label = {0: 'Negative',1: 'Neutral', 2: 'Positive'}

        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
        sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL)
        config = AutoConfig.from_pretrained(MODEL)
        config.update({"id2label": id2label})

        device = f"cuda" if torch.cuda.is_available() else "cpu"  #
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL, config=config).to(device)
        top_texts = [ 'A beautiful well-appointed kitchen with a counter window.', 'three ugly mugs are on the kitchen counter.',]
        inputs = sentiment_tokenizer(top_texts, padding=True, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        logits = sentiment_model(**inputs)['logits']
        prob = nn.functional.softmax(logits, dim=-1)
        output_dir_for_finetuned_roberta = os.path.join(os.path.expanduser('~'),'checkpoints','finetuned_robert3')
        finetuned_roberta_model_path = os.path.join(output_dir_for_finetuned_roberta,'pytorch_model.bin')
        finetuned_roberta_config = os.path.join(output_dir_for_finetuned_roberta,'config.json')

        config2 = AutoConfig.from_pretrained(finetuned_roberta_config)
        sentiment_model2 = AutoModelForSequenceClassification.from_pretrained(finetuned_roberta_model_path, config=config2).to(device)
        logits2 = sentiment_model2(**inputs)['logits']
        prob3 = nn.functional.softmax(logits2, dim=-1)

    ##############
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL)
    output_dir_for_finetuned_roberta = os.path.join(os.path.expanduser('~'),'checkpoints','finetuned_roberta')
    finetuned_roberta_model_path = os.path.join(output_dir_for_finetuned_roberta,'pytorch_model.bin')
    finetuned_roberta_config = os.path.join(output_dir_for_finetuned_roberta,'config.json')
    annotations_path = '/Users/danielabendavid/data/senticap/annotations'
    annotations_path = replace_user_home_dir(annotations_path)

    train_texts, train_labels = get_annotations_data(annotations_path, 'train')
    val_texts, val_labels = get_annotations_data(annotations_path, 'val')
    test_texts, test_labels = get_annotations_data(annotations_path, 'test')

    id2label = {0: 'Negative',1: 'Neutral', 2: 'Positive'}
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    train_labels = [label_mapping[label] for label in train_labels]
    val_labels = [label_mapping[label] for label in val_labels]
    test_labels = [label_mapping[label] for label in test_labels]

    # Create a Dataset object
    train_dataset2 = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset2 = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_dataset2 = Dataset.from_dict({"text": test_texts, "label": test_labels})

    def sentiment_tokenize(batch):
        return sentiment_tokenizer(batch["text"], padding=True, truncation=True, max_length=256)

    train_dataset2 = train_dataset2.map(sentiment_tokenize, batched=True, batch_size=len(train_dataset2))
    val_dataset2 = val_dataset2.map(sentiment_tokenize, batched=True, batch_size=len(val_dataset2))
    test_dataset2 = test_dataset2.map(sentiment_tokenize, batched=True, batch_size=len(test_dataset2))

    # Set dataset format
    train_dataset2.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset2.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset2.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # We will need this to directly output the class names when using the pipeline without mapping the labels later.
    # Extract the number of classes and their names
    num_labels = len(set(train_labels))
    class_names = list(set(train_labels))
    print(f"number of labels: {num_labels}")
    print(f"the labels: {class_names}")


    # Update the model's configuration with the id2label mapping
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    config = AutoConfig.from_pretrained(MODEL)
    config.update({"id2label": id2label})
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL, config=config)
    # model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)
    train_dataset = train_dataset2
    val_dataset = val_dataset2
    test_dataset = test_dataset2
    ############


    # model, train_dataset, val_dataset = use_source_tutorial()


    # TrainingArguments
    training_args = TrainingArguments(
        output_dir='./results', # repository_id,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir='./logs', #f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="tensorboard",
        # push_to_hub=True,
        # hub_strategy="every_save",
        # hub_model_id=repository_id,
        # hub_token=HfFolder.get_token(),
    )

    # Trainer
    trainer = Trainer(
        model=sentiment_model, #model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        )


    print("start to train...")
    trainer.train()
    trainer.evaluate()

    # Save the model
    trainer.save_model(output_dir_for_finetuned_roberta)
    print(f"Model saved to: {output_dir_for_finetuned_roberta}")

    # TEST MODEL
    ################
    device = f"cuda" if torch.cuda.is_available() else "cpu"

    top_texts = [ 'A beautiful well-appointed kitchen with a counter window.', 'three ugly mugs are on the kitchen counter.',]
    inputs = sentiment_tokenizer(top_texts, padding=True, return_tensors="pt")
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    logits = sentiment_model(**inputs)['logits']
    prob = nn.functional.softmax(logits, dim=-1)

    config2 = AutoConfig.from_pretrained(finetuned_roberta_config)
    sentiment_model2 = AutoModelForSequenceClassification.from_pretrained(finetuned_roberta_model_path, config=config2).to(device)
    logits2 = sentiment_model2(**inputs)['logits']
    prob2 = nn.functional.softmax(logits, dim=-1)


    #check accuracy
    # Define a function to compute the predicted labels
    def compute_predictions(predictions):
        predicted_labels = []
        for logits in predictions:
            predicted_label = int(np.argmax(logits))
            predicted_labels.append(predicted_label)
        return predicted_labels

    # Perform the predictions
    check_dataset = test_dataset
    check_labels = test_labels
    predictions = trainer.predict(check_dataset)
    predicted_labels = compute_predictions(predictions.predictions)

    # Compute accuracy
    correct_predictions = sum(pred == true_label for pred, true_label in zip(predicted_labels, check_labels))
    accuracy = correct_predictions / len(check_labels)

    # Print accuracy
    print(f"Accuracy: {accuracy}")

    #####
    for text in test_texts:
        if text in train_texts:
            print("bug")
    #####
    sentiment_grades = None
    if sentiment_type == 'positive':
        sentiment_grades = nn.functional.softmax(logits, dim=-1)[:, 2]
    elif sentiment_type == 'neutral':
        sentiment_grades = nn.functional.softmax(logits, dim=-1)[:, 1]
    elif sentiment_type == 'negative':
        sentiment_grades = nn.functional.softmax(logits, dim=-1)[:, 0]
    sentiment_grades = sentiment_grades.unsqueeze(0)

    ################
    from transformers import pipeline

    finetuned_roberta_model_path
    classifier = pipeline('text-classification', finetuned_roberta_config)
    classifier = pipeline('text-classification', repository_id)

    text = "Kederis proclaims innocence Olympic champion Kostas Kederis today left hospital ahead of his date with IOC inquisitors claiming his innocence and vowing: quot;After the crucifixion comes the resurrection. quot; .."
    result = classifier(text)

    predicted_label = result[0]["label"]
    print(f"Predicted label: {predicted_label}")
    print("finish")

if __name__=='__main__':
    main()
