import html
import re

import nltk
import torchtext
import torch


re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
    'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
    '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
    ' @-@ ','-').replace('\\', ' \\ ').replace('\u200d','').replace('\xa0',' ').replace(
    '\u200c','').replace('“',' ').replace('”',' ').replace('"',' ').replace('\u200b','')
    x = re.sub('[\(\[].*?[\)\]]', '', x)
    x = re.sub('<[^<]+?>', '', x)
    x = re.sub('[A-Za-z]+','ENG ', x)
    x = re.sub(r'\d+.?(\d+)?','NUM ',x).replace("(","").replace(")","")
    return re1.sub(' ', html.unescape(x))

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = []
    for token in raw_html:
        cleantext_tmp = re.sub(cleanr, ' ', token)
        cleantext.append(fixup(cleantext_tmp))
    return cleantext

def preprocessing(train_path: str, test_path: str, embedding_path: str, minimum_freq: int = 2,max_sent_len: int = 100,save="models/word2idx.pkl"):
    """ Data Loading and Preprocessing"""
    text = torchtext.data.Field(sequential=True, use_vocab=True, lower=False,tokenize=nltk.word_tokenize,init_token="</bos>",eos_token="</eos>",preprocessing=cleanhtml, batch_first=True,is_target=False, fix_length=max_sent_len,include_lengths=True)
    target = torchtext.data.Field(sequential=False, use_vocab=False,batch_first=True, is_target=True)
    dataset = torchtext.data.TabularDataset(train_path, format='csv',fields={"target": ('target', target), "text": ('text', text)})
    data_test = torchtext.data.TabularDataset(test_path, format='csv',fields={"target": ('target', target), "text": ('text', text)})
    
    # Build Vocab
    text.build_vocab(dataset, data_test, min_freq=minimum_freq)
    text.vocab.load_vectors(torchtext.vocab.Vectors(embedding_path))
    vocab_size = len(text.vocab.itos)
    padding_idx = text.vocab.stoi[text.pad_token]
    
    # Split Train Data to train, validation sets
    data_train, data_val = dataset.split(split_ratio=0.9)
    torch.save(dict(text.vocab.stoi),save)

    print("train set size:", len(data_train))
    print("val set size:", len(data_val))
    print("test set size:", len(data_test))
    print("vocab size:", len(text.vocab.itos))
    print("embed shape:", text.vocab.vectors.shape)
    print('')
    args_dict = {
        "data_train": data_train, "data_val": data_val,
        "data_test": data_test, "vocab_size": vocab_size,"pretrained_embeddings":text.vocab.vectors,
        "padding_idx": padding_idx}

    return args_dict