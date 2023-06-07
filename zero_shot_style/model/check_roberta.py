import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer # SENTIMENT


def preprocess(text):
    def preprocess_single_text(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    if type(text) == list:
        new_text_list = []
        for t in text:
            new_text_list.append(preprocess_single_text(t))
        return new_text_list
    else:
        return preprocess_single_text(text)

def main():
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    sentiment_model_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    sentiment_model.to(device)
    sentiment_model.eval()
    for param in sentiment_model.parameters():
        param.requires_grad = False
    # SENTIMENT: tokenizer for sentiment analysis module
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_type = 'positive'
    clip_loss_temperature = 0.01

    top_texts = ['beautifull girl', 'lazy girl']
    top_texts = ['we are going to boring trip tomorrow', 'we are going to trip tomorrow', 'we are going to beautifull trip tommorrow']
    # ######todo: daniela debug    effect of update CLIP
    # # top_texts = ["The bedroom used child abuse"]*DEBUG_NUM_WORDS+["The bedroom of a sweet baby"]*DEBUG_NUM_WORDS
    # for i in range(len(top_texts)):
    #     if i<=len(top_texts)/2:
    #         top_texts[i] = "The bedroom used child abuse"
    #     else:
    #         top_texts[i] = "The bedroom of a sweet baby"
    # ######todo: daniela debug    effect of update CLIP
    # get score for text
    with torch.no_grad():
        text_list = preprocess(top_texts)
        encoded_input = sentiment_tokenizer(text_list, padding=True, return_tensors='pt').to(device)
        output = sentiment_model(**encoded_input)
        scores = output[0].detach()
        scores1 = nn.functional.softmax(scores, dim=-1)
        scores2 = nn.functional.softmax(scores1, dim=0)
        # sentiment_grades = None
        if sentiment_type == 'positive':
            sentiment_grades = scores2[:, 2]
        elif sentiment_type == 'neutral':
            sentiment_grades = scores2[:, 1]
        elif sentiment_type == 'negative':
            sentiment_grades = scores2[:, 0]
        sentiment_grades = sentiment_grades.unsqueeze(0)

        predicted_probs = nn.functional.softmax(sentiment_grades / clip_loss_temperature, dim=-1).detach()
    print('finish')

if __name__=='__main__':
    main()
    print('finish')
