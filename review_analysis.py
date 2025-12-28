from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from model_utils import predict, sentiment_map, aspect_map

def analyze_reviews(reviews, include_general=False):
    aspects = []
    sentiment_count = {"Positive": 0, "Negative": 0}
    for review in reviews:
        lines = sent_tokenize(review)
        for line in lines:
            line = line.strip()
            if len(line.split()) > 3:
                s, a = predict(line)
                s_label = sentiment_map[s]
                a_label = aspect_map[a]
                if include_general or a_label != "general":
                    aspects.append((a_label, s_label, line))
                    sentiment_count[s_label] += 1
    return aspects, sentiment_count


def score_hospital(h):
    return 0.3 * (1 - h["distance_km"] / 10) + 0.3 * h["positive_ratio"] + 0.4 * (len(h["aspect_summary"]) / 7)
