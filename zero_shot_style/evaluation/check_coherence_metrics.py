from evaluate import load

import nltk
from nltk.util import ngrams
import math

def perplexity(model, test_data):
    perplexity = 0
    count = 0
    for sentence in test_data:
        sentence_prob = model.entropy(sentence)
        perplexity += sentence_prob
        count += 1
    perplexity = perplexity / count
    perplexity = 2 ** perplexity
    return perplexity

import gensim
from gensim.models import CoherenceModel

def coherence_score(model, texts, dictionary, corpus):
    coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()


import textstat


def readability_scores(text):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "smog_index": textstat.smog_index(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
        "difficult_words": textstat.difficult_words(text),
        "linsear_write_formula": textstat.linsear_write_formula(text),
        "gunning_fog": textstat.gunning_fog(text),
        "text_standard": textstat.text_standard(text)
    }
def f1(tests):
    # gpt2_perplexity
    model_id = 'gpt2'
    perplexity = load("perplexity", module_type="measurement")
    results = perplexity.compute(data=tests, model_id=model_id, add_start_token=True)

def f2():
    from nltk.lm import MLE
    train_data = ["hwllo hhow are you,dsjgdaskd dasjkgdajsjkgdad sadjgasjdasdaDJ,DASS  DASJHKDHASKLD;d"]
    model = MLE(3)  # trigram model
    model.fit(train_data)  # train_data is a list of lists of words

    # Calculate perplexity on test data
    test_data = [["this", "is", "a", "sentence"], ["another", "sentence"]]
    print("Perplexity:", perplexity(model, test_data))

def f3(texts):
    # Train a topic model
    from gensim.models import LdaModel
    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)

    # Calculate coherence score
    print("Coherence Score:", coherence_score(model, texts, dictionary, corpus))

def f4():
    # Calculate readability scores for a given text
    text = "This is an example sentence for testing readability scores."
    scores = readability_scores(text)

    # Print readability scores
    print("Flesch Reading Ease:", scores["flesch_reading_ease"])
    print("SMOG Index:", scores["smog_index"])
    print("Flesch-Kincaid Grade:", scores["flesch_kincaid_grade"])
    print("Coleman-Liau Index:", scores["coleman_liau_index"])
    print("Automated Readability Index:", scores["automated_readability_index"])
    print("Dale-Chall Readability Score:", scores["dale_chall_readability_score"])
    print("Difficult Words:", scores["difficult_words"])
    print("Linsear Write Formula:", scores["linsear_write_formula"])
    print("Gunning Fog:", scores["gunning_fog"])
    print("Text Standard:", scores["text_standard"])


def main():
    tests = ["how are you", "The wonderful waves.", "he is a table", "amhge asjkgdeaj ddjas"]
    # f1()
    # f2()
    # f3()
    f4()
    print("finish")

if __name__=='__main__':
    main()