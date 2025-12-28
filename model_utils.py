import os
import re
import torch
from huggingface_hub import hf_hub_download
from transformers import BertTokenizer
from model import MultiTaskBert

sentiment_map = {0: "Negative", 1: "Positive"}
aspect_map = {
    0: "cleanliness", 1: "cost", 2: "emergency",
    3: "general", 4: "staff", 5: "treatment", 6: "waiting_time"
}

_model_cache = None

def load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiTaskBert()

        local_model_path = os.path.join("models", "multitask_bert_model.pth")
        model_path = None
        if os.path.exists(local_model_path):
            model_path = local_model_path
        else:
            try:
                model_path = hf_hub_download(repo_id="amit2005/sentimed-model", filename="multitask_bert_model.pth")
            except Exception:
                model_path = None

        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        else:
            model = None

        try:
            tokenizer = BertTokenizer.from_pretrained("models/bert_tokenizer", local_files_only=True)
        except Exception:
            try:
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            except Exception:
                tokenizer = None

        if model is not None and tokenizer is not None:
            _model_cache = (model, tokenizer, device)
        else:
            _model_cache = None
        return _model_cache
    except Exception:
        _model_cache = None
        return _model_cache

def predict(text):
    cache = load_model()
    if cache is not None:
        model, tokenizer, device = cache
        inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            s_logits, a_logits = model(input_ids, attention_mask)
        sentiment = torch.argmax(s_logits, dim=1).item()
        aspect = torch.argmax(a_logits, dim=1).item()
        return sentiment, aspect

    pos_words = [
        "good", "great", "excellent", "clean", "friendly", "helpful", "quick", "efficient", "best"
    ]
    neg_words = [
        "bad", "poor", "rude", "dirty", "slow", "long", "expensive", "worst", "delay"
    ]
    text_l = text.lower()
    p = sum(text_l.count(w) for w in pos_words)
    n = sum(text_l.count(w) for w in neg_words)
    sentiment = 1 if p >= n else 0

    aspect_keywords = {
        0: ["clean", "hygiene", "sanit"],
        1: ["cost", "price", "expensive", "bill", "fees"],
        2: ["emergency", "ambulance", "urgent"],
        4: ["staff", "nurse", "doctor", "reception"],
        5: ["treatment", "care", "therapy", "surgery"],
        6: ["wait", "waiting", "delay", "queue"],
    }
    aspect = 3
    for idx, keys in aspect_keywords.items():
        if any(k in text_l for k in keys):
            aspect = idx
            break

    return sentiment, aspect
