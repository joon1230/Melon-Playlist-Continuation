import pandas as pd
import KakaoArena.Filter_date as fdate
import KakaoArena.Merged_table as mt # 후보군 테이블 merge
import KakaoArena.Clustering_km as km
import KakaoArena.Get_df as gd
import KakaoArena.Get_cluster_matrix as gcm
import os
import time
print( os.getcwd() )

#%%

# 데이터 불러오기


genre = pd.read_json('data/genre_gn_all.json', typ = 'series', encoding = "utf-8")
meta = pd.read_json('data/song_meta.json', typ = 'frame', encoding="utf-8")
train = pd.read_json('data/train.json', typ = 'frame',encoding="utf-8")
val = pd.read_json('data/val.json', typ = 'frame',encoding="utf-8")
test = pd.read_json('data/test.json', typ = 'frame',encoding="utf-8")

## FLAG
complex_ = pd.concat([ train , val] , axis = 0 )
complex_ = pd.concat([ complex_ , test ] , axis = 0 )

#%% preprocessing part!!!
"""
전처리 구간. 
이상치 관련 : 음원 날짜 수정 # 
형태소 분석 : 플레이리스트 제목 형태소 분석 # id_title
"""
print(os.getcwd())
start = time.time()

# meta data  date 필터링 해주기
print( "%-20s"%("date filtering"))
filter_date = fdate.Filter_date( complex_ , meta )
meta = filter_date.kill_zeros()

merged_t = mt.Merged_table( complex_, genre, meta )
get_df = gd.Get_data(complex_, genre, meta)


# tokenize
print( "%-20s"%("tokenize"), end = '\r')
tokenized_title = get_df.get_title_df()
tokenized_title.to_pickle("data/preprocessed/tokenized_title.pickle")

print( (time.time() - start)/60 )

#%%
"""
데이터 수치화 구간
song 
tag & genre & title
tag
genre
album
"""
clu_km = km.Clustering( complex_, meta )

song_clu = clu_km.clustering( by = 'songs'  )
song_clu.to_pickle("data/digitization/clu_song_emb_200.pickle")
del song_clu
#
tag_gnr_title_clu = clu_km.clustering( merged_t.tag_gnr_title(id_title = tokenized_title) , by = 'tag_gnr_title')
tag_gnr_title_clu.to_pickle("data/digitization/clu_tag_gnr_title_emb_100.pickle")
del tag_gnr_title_clu

# C. singer
singer_clu = clu_km.clustering(merged_t.title_singer(id_title = tokenized_title) , by = 'title_singer')
singer_clu.to_pickle("data/digitization/clu_singer_emb_100.pickle")
del singer_clu

# D. tag : complex_
tag_clu = clu_km.clustering( by = 'tags'  )
tag_clu.to_pickle("data/digitization/clu_tag_emb_30.pickle")
del tag_clu

# E. dtl_genre 원핫 인코딩을 했으므로 이 단계 생략

# F . album  아래에서 df 랑 clustering이랑 코드가 합쳐져있음.
emb_df, album_df = clu_km.clustering( by = 'album'  )
emb_df.to_pickle('data/digitization/clu_album_emb100.pickle' )
del emb_df
album_df.to_pickle('data/preprocessed/album_df_.pickle') # 데이타 프레임 윗단계

#%%
"""
make clustering matrix
"""
matrix = gcm.Get_cluster_matrix( complex_, meta, genre )

# songs
song_vec = pd.read_pickle( 'data/digitization/clu_song_emb_200.pickle')
song_vec = song_vec[['songs_id' , 'label'] + list(song_vec.columns[3:12])]
train_song = matrix.get_cluster_matrix( train , song_vec , by = "song")
val_song = matrix.get_cluster_matrix( val , song_vec , by = "song")

train_song.to_csv('data/matrix/train/train_song.csv' , index = False)
val_song.to_csv('data/matrix/val/val_song.csv' , index = False)

# word_tag_gnr_title
word_vec = pd.read_pickle( 'data/digitization/clu_tag_gnr_title_emb_100.pickle' )
total_word = matrix.get_cluster_matrix( merged_t.tag_gnr_title() , word_vec , by = 'word')
val_word = total_word.loc[ total_word.id.isin( list(val.id.values) ) ] # word ( val )
train_word = total_word.loc[ total_word.id.isin( list(train.id.values ) ) ] # word

val_word.to_csv("data/matrix/val/val_word.csv" , index = False)
train_word.to_csv("data/matrix/train/train_word.csv" , index = False)

# singer
singer_vec = pd.read_pickle( 'data/digitization/clu_singer_emb_100.pickle')
total_singer = matrix.get_cluster_matrix( merged_t.title_singer() , singer_vec , by = 'word')
val_singer = total_singer.loc[ total_singer.id.isin( list(val.id.values) ) ] # singer
train_singer = total_singer.loc[ total_singer.id.isin( list(train.id.values) ) ] # singer

val_singer.to_csv("data/matrix/val/val_singer.csv" , index = False)
train_singer.to_csv("data/matrix/train/train_singer.csv" , index = False)

# tags
tag_df = pd.read_pickle('tag_df.pickle') # clustring_tag 함수를 실행시키면 자동으로 저장됨
tag_vec = pd.read_pickle('data/digitization/clu_tag_emb_30.pickle') # train , val , test 모두 합쳐서 embeding
total_tag = matrix.get_cluster_matrix( tag_df , tag_vec , by = 'tag')
val_tag = total_tag.loc[ total_tag.id.isin( list(val.id.values) )]
train_tag = total_tag.loc[ total_tag.id.isin( list(train.id.values) )]

val_tag.to_csv("data/matrix/val/val_tag.csv" , index = False)
train_tag.to_csv("data/matrix/train/train_tag.csv" , index = False)

# genre
total_gnr = matrix.get_genre_matrix( complex_ ,meta, genre )
val_gnr = total_gnr.loc[ total_gnr.id.isin( list(val.id.values) ) ]
train_gnr = total_gnr.loc[ total_gnr.id.isin( list(train.id.values) ) ]

val_gnr.to_csv("data/matrix/val/val_gnr.csv" , index = False)
train_gnr.to_csv("data/matrix/train/train_gnr" , index = False)

# album
album_df = pd.read_pickle('data/preprocessed/album_df.pickle')
album_vec = pd.read_pickle('data/clu_album_emb100.pickle')
total_album = matrix.get_cluster_matrix( album_df , album_vec , by = 'album')
train_album = total_album.loc[ total_album.id.isin( train.id.values ) ]
val_album = total_album.loc[ total_album.id.isin( val.id.values ) ]

train_album.to_csv("data/matrix/train/train_album.csv",index=False)
val_album.to_csv("data/matrix/val/val_album.csv",index=False)

# plylst_all
train_ply = pd.merge( train_song , train_word , on = 'id' , how = 'inner')
train_ply = pd.merge( train_ply , train_singer , on = 'id' , how = 'inner')
train_ply = pd.merge( train_ply , train_gnr , on = 'id' , how = 'inner')
train_ply = pd.merge( train_ply , train_tag , on = 'id' , how = 'inner')
train_ply = pd.merge( train_ply , train_album , on = 'id' , how = 'inner')

val_ply = pd.merge( val_song , val_word , on = 'id' , how = 'inner')
val_ply = pd.merge( val_ply , val_singer , on = 'id' , how = 'inner')
val_ply = pd.merge( val_ply , val_gnr , on = 'id' , how = 'inner')
val_ply = pd.merge( val_ply , val_tag , on = 'id' , how = 'inner')
val_ply = pd.merge( val_ply , val_album , on = 'id' , how = 'inner')



#%%
print(train_song.head())