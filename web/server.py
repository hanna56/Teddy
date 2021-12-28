# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np

import kss
from googletrans import Translator
from itertools import combinations

from krwordrank.word import summarize_with_keywords
from konlpy.tag import Mecab 
mecab=Mecab(dicpath=r"/mecab/mecab-ko-dic")

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

app = Flask(__name__)

with open('korean_stopwords.txt', 'r', encoding='UTF8') as f:
		stopwords = f.read().split()

def extract_keyword(text): # 텍스트에서 명사/대명사만 추출 후 모델 실행
    keyword_list=[]
    
    nouns=[w[0] for w in mecab.pos(text) if w[1]=="NNP" or w[1]=="NNG"]
    noun_combine = ' '.join(nouns)

    #하이퍼파라미터 조정하기 # 짧은글에선 min_count=1, 긴글에선 min_count=3
    try: 
        keywords = summarize_with_keywords([noun_combine], min_count=3, max_length=10,
        beta=0.85, max_iter=10, stopwords=stopwords, verbose=True)

    except:
        # print("<except 실행>")
        keywords = summarize_with_keywords([noun_combine], min_count=1, max_length=10,
        beta=0.85, max_iter=10, stopwords=stopwords, verbose=True)

    for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:5]:
        keyword_list.append(word)

    return keyword_list

bertmodel, vocab = get_pytorch_kobert_model()

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# 기본 Bert tokenizer 사용
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair) 

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 5, # softmax 사용 <- binary일 경우는 2
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'


@app.route("/")
def index():
    return render_template('main.html')

@app.route('/findemo',methods = ['GET', 'POST'])
def getFindemo():
    if request.method =='GET':
        return render_template('findemo.html')
    elif request.method == 'POST':
        input_text = request.form['input_text']
        
        text = input_text
        output_text = input_text
        text=kss.split_sentences(text)

        keywords = extract_keyword(input_text)

        # translate the keyword for image search
        keywords_eng = []
        translator = Translator()
        for word in keywords:
            eng_word = translator.translate(word, dest = 'en')
            keywords_eng.append(eng_word.text)

        # keyword combinations
        comb = combinations(keywords_eng, 2)

        test_sen=[]
        text_lst=[]

        for i in range(len(text)):
            text_lst=[text[i]]
            text_lst.append('1')
            test_sen.append(text_lst)

        # Setting parameters
        max_len = 128 # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
        batch_size = 32

        data_test_sen = BERTDataset(test_sen, 0, 1, tok, max_len, True, False)
        test_sen_dataloader = torch.utils.data.DataLoader(data_test_sen, batch_size=batch_size, num_workers=0)

        model = BERTClassifier(bertmodel, dr_rate=0.5)
        model.load_state_dict(torch.load("full_model.pt", map_location=map_location))
        model.eval()

        # Setting parameters
        max_len = 128 # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
        batch_size = 32
    
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_sen_dataloader):
            token_ids = token_ids.long()
            segment_ids = segment_ids.long()
            valid_length= valid_length
            label = label.long()
            out = model(token_ids, valid_length, segment_ids)
        print(out)

        # {0: '공포', 1: '기쁨', 2: '분노', 3: '사랑', 4: '슬픔'}
        score=[0,0,0,0,0]
        out_lst=out.tolist()

        for i in range(len(out_lst)):
            score[out_lst[i].index(max(out_lst[i]))]+=1
        print(score)

        result =''
        # {0: '공포', 1: '기쁨', 2: '분노', 3: '사랑', 4: '슬픔'}
        if score.index(max(score))==0:
            result = "공포"
        elif score.index(max(score))==1:
            result = "기쁨"
        elif score.index(max(score))==2:
            result = "분노"
        elif score.index(max(score))==3:
            result = "사랑"
        elif score.index(max(score))==4:
            result = "슬픔"
    
        return render_template('result.html', Result = result, Output_text = output_text, keywords = keywords_eng, score = score, combi = comb)

@app.route('/aboutus')
def getAboutus():
    return render_template('about_us.html')

@app.route('/result',methods = ['GET', 'POST'])
def getResult():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug = True)
