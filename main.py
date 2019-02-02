import torch
import torchtext
from tqdm import tqdm

from model import MalayalamModel
from preprocess import preprocessing
from sklearn.metrics import f1_score,accuracy_score


def save_model(model, model_path):
    """Save model."""
    torch.save(model.state_dict(), model_path)

def load_model(model, model_path, use_cuda=False):
    """Load model."""
    map_location = 'cpu'
    if use_cuda and torch.cuda.is_available():
        map_location = 'cuda:0'
    model.load_state_dict(torch.load(model_path, map_location))
    return model

data_dict = preprocessing("data/train.csv","data/test.csv","embeddings/malayalam200.txt")

train_data = data_dict["data_train"]
val_data = data_dict["data_val"]
test_data = data_dict["data_test"]

classifier = MalayalamModel(data_dict["pretrained_embeddings"],data_dict["padding_idx"])

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'

batch_size = 256
epochs = 20
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)

classifier = classifier.to(device)

train_iter = torchtext.data.Iterator(dataset=train_data, batch_size=batch_size,train=True, shuffle=True, sort=False,device=device)
val_iter = torchtext.data.Iterator(dataset=val_data, batch_size=batch_size*2,train=True, shuffle=True, sort=False,device=device)
test_iter = torchtext.data.Iterator(dataset=test_data, batch_size=batch_size*2,train=True, shuffle=True, sort=False,device=device)

for epoch in range(epochs):
    classifier.train()
    train_iter.init_epoch()
    train_predict = []
    train_true = []
    for batch in tqdm(train_iter):
            (text, text_len), target = batch.text, batch.target
            train_true.extend(target.to('cpu').numpy().tolist())
            optimizer.zero_grad()
            logits,prediction = classifier(text, text_len)
            train_predict.extend(torch.argmax(prediction,dim=1).to('cpu').numpy().tolist())
            loss = classifier.loss(logits, target)
            loss.backward()
            optimizer.step()
    print("Training Accuracy : ",accuracy_score(train_true,train_predict))
    val_predict = []
    val_true = []
    classifier.eval() 
    for batch in tqdm(val_iter):
        (text, text_len), target = batch.text, batch.target
        with torch.no_grad():
            val_true.extend(target.to('cpu').numpy().tolist())
            _,prediction = classifier(text,text_len)
            val_predict.extend(torch.argmax(prediction,dim=1).to('cpu').numpy().tolist())
    print("Validation Accuracy : ",accuracy_score(val_true,val_predict))
    test_predict = []
    test_true = []
    classifier.eval() 
    for batch in tqdm(test_iter):
        (text, text_len), target = batch.text, batch.target
        with torch.no_grad():
            test_true.extend(target.to('cpu').numpy().tolist())
            _,prediction = classifier(text,text_len)
            test_predict.extend(torch.argmax(prediction,dim=1).to('cpu').numpy().tolist())
    print("testing Accuracy : ",accuracy_score(test_true,test_predict))