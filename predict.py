import torch
from model import MalayalamModel
from preprocess import cleanhtml
from nltk import word_tokenize
import numpy as np


def load_model(model, model_path, use_cuda=False):
    """Load model."""
    map_location = 'cpu'
    if use_cuda and torch.cuda.is_available():
        map_location = 'cuda:0'
    model.load_state_dict(torch.load(model_path, map_location))
    return model

word2idx = torch.load("models/word2idx.pkl",'cpu')

idx2label = np.load("models/label2idx.npy").item()
label2idx = dict(zip(idx2label.values(),idx2label.keys()))

classifier = MalayalamModel(pretrained_embed=False,padding_idx=1)

classifier = load_model(classifier,"models/model.pkl")

def classify(text: str):
    tokens = word_tokenize(text)
    tokens = cleanhtml(tokens)
    fixed_text = " ".join(tokens)
    tokens = [word2idx.get(token,0) for token in tokens]
    tokens = torch.tensor(tokens).expand(1,-1)
    seq_len = torch.tensor([len(tokens)])
    _,labels = classifier(tokens,seq_len)
    labels = torch.softmax(labels,dim=1)
    labels = labels.detach().numpy().tolist()[0]
    # import ipdb; ipdb.set_trace()
    intent_ranking = []
    intent = {}
    for i, out in enumerate(labels):
            temp = {"confidence": float(format(out, 'f')), "name": label2idx[i]}
            intent_ranking.append(temp)
    intent_ranking = sorted(intent_ranking, key=lambda e: e['confidence'], reverse=True)
    intent.update({
                "intent": intent_ranking.pop(0),
                "intent_ranking": intent_ranking
    })
    intent.update({"processed_text": fixed_text})
    return intent