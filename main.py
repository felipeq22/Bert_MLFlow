import torch
#import tensorflow
import mlflow
import mlflow.pytorch
from transformers import BertTokenizer, BertForSequenceClassification

mlflow.start_run()

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

input_text = 'This is a sample text for Bert'

inputs = tokenizer(input_text, return_tensors = 'pt',
                   padding= True, truncation = True,
                   max_length=64
                   )

outputs =model(**inputs)

logits = outputs.logits
predicted_probabilites = torch.softmax(logits, dim = 1)
print(predicted_probabilites)

mlflow.pytorch.log_model(model, "bert_model")
mlflow.log_params({
    'model_name':model_name,
    'input_text': input_text
})

mlflow.log_metrics({
    "class_0_probability": predicted_probabilites[0][0].item(),
    "class_1_probability": predicted_probabilites[0][1].item()
})

mlflow.end_run()
