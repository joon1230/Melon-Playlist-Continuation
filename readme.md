# 플레이리스트에 가장 어울리는 곡들을 예측할 수 있을까?

플레이리스트에 있는 곡들과 비슷한 느낌의 곡들을 계속해서 듣고 싶은 적이 있으셨나요?

이번 대회에서는 플레이리스트에 수록된 곡과 태그의 절반 또는 전부가 숨겨져 있을 때, 주어지지 않은 곡들과 태그를 예측하는 것을 목표로 합니다.

만약 플레이리스트에 들어있는 곡들의 절반을 보여주고, 나머지 숨겨진 절반을 예측할 수 있는 모델을 만든다면, 플레이리스트에 들어있는 곡이 전부 주어졌을 때 이 모델이 해당 플레이리스트와 어울리는 곡들을 추천해 줄 것이라고 기대할 수 있을 것입니다.

---
1. 이상치 처리
    플레이리스트와 일치하지 않는 노래들의 발매일 수정( 가장 최신의 플레이리스트의 날짜로 수정 .. )
    
requirment
```
python 3.6
sentenspiece
gensim
sklearn
numpy 
pandas
tqdm
```

-----
수치화 items matrix
- tag_gnr_title ( words )
- title_singer ( singer )
- tag ( tag )
- song ( song )
- genre ( genre )
- album ( album )

---- 
데이터 전처리 
```angular2
$ python preprocessing.py
```
result
```angular2
file_lst..
```


