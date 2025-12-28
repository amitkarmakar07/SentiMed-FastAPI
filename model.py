import torch
from torch import nn
from transformers import BertModel

class MultiTaskBert(nn.Module):
    def __init__(self):
        super(MultiTaskBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.sentiment_classifier = nn.Linear(768, 2)  # Not Sequential
        self.aspect_classifier = nn.Linear(768, 7)     # Not Sequential

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        sentiment_logits = self.sentiment_classifier(pooled)
        aspect_logits = self.aspect_classifier(pooled)
        return sentiment_logits, aspect_logits
