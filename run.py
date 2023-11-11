from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

def preprocess_tweet(tweet):
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    return " ".join(tweet_words).lower()

def perform_sentiment_analysis(tweet, model, tokenizer, labels):
    tweet_proc = preprocess_tweet(tweet)

    # Sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)

    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)

    for label, score in zip(labels, scores):
        rounded_score = round(score)
        print(label, rounded_score)

if __name__ == "__main__":
    # Load model and tokenizer
    load_model = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(load_model)
    tokenizer = AutoTokenizer.from_pretrained(load_model)

    labels = ['Negative', 'Neutral', 'Positive']

    # Analyze sentiment for a tweet
    tweet = 'what a bad day'
    perform_sentiment_analysis(tweet, model, tokenizer, labels)
