# encoding: utf-8
import pandas as pd 
import numpy as np
from krwordrank.hangle import normalize
from krwordrank.word import summarize_with_keywords
from konlpy.tag import Mecab 
mecab=Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

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

# text="공포소설 추천책 <그 환자>는 독자를 철저하게 혼돈스럽게 만드는데소설이지만 이건 100% 실화일 것이라는생각이 들 정도로 문장의 묘사가 섬세하기 때문입니다.그러나 실제 있었던 일이라고 생각이 드는 순간, 며칠 간 잠을 못 아루게 되는무섭고 소름끼치는 이야기가 담겨있습니다.그래서 오히려 저는 읽는 내내이건 소설이야.. 꾸며낸 이야기여야만 해..라고 스스로 되뇌면서 읽었던 것 같네요.그렇제 않고 실제로 있었던 일이라는생각이 드는 순간두려움에 이성의 끈을 놓칠 것 같았거든요.이 이야기가 오죽 무서웠으면미국의 가장 큰 커뮤니티인 '레딧'이라는사이트 공포 게시판에 연재 되었었는데사람들의 폭발적인 반응으로베스트 게시물이 되었다고 하네요.국내에서도 한 아마추어 번역가가연재글을 번역해서 올린 적이 있었는데다음 편이 언제 나오냐는 댓글이 폭발하기도 했죠. 이 장르는 사실 마니아칙하고 마이너한 느낌이라많은 분들이 찾지 않으시는데 공포소설 추천을해드리는 이유는 <그 환자>는 미스테리, 공포, 스릴러가 결합되어 있는 듯한 높은 퀄리티의좋은 작품이기 때문입니다.단순히 놀래키는 장치들만 나오는 것이 아니라소재 자체가 참신해서 보는 재미가 있었네요.정신병원의 한 환자 이야기를 소재로 선택한재스퍼 드윗은 완벽하게 탄탄한 스토리로얼마나 그의 역량이 대단한지를데뷔작으로 증명해냈습니다."
# text = "'적신호' 켜진 2학기 전면등교…등교선택권·시차등교제가 대안될까"
# print(extract_keyword(text))