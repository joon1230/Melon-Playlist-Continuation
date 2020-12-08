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


%#@$$# 
get_df.get_title_df 쑤시기 (형태소 분할된 데이터 들이 남아 있게끔...)

Merged_table.tag_gnr_title ( 마찬가지로 형태소 분할된 데이터들을 재활용 할수 있게끔... )

++ 형태소 분할은 pre... 부분에 들어 갈 수 있게 수정 .. 거기에 맞춤형 데이터 프레임 생성??..

