import pandas as pd
import KakaoArena.Filter_date as fdate
import KakaoArena.Merged_table as mt # 후보군 테이블 merge
import KakaoArena.Clustering_km as km
import KakaoArena.Get_df as gd
import KakaoArena.Get_cluster_matrix as gcm
import KakaoArena.Select_candidate as sc
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

#%%

# SAMPLE
start = time.time()

# meta data  date 필터링 해주기
filter_date = fdate.Filter_date( complex_ , meta )
meta = filter_date.kill_zeros()

print( (time.time() - start)/60 )
# meta.to_pickle("data/done_filter_meta.pickle")
# meta = pd.read_pickle( 'data/done_filter_meta.pickle') # 한번했으면 이것만 불러오기


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
id_title = get_df.get_title_df()

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

#song_clu = clu_km.clustering_test( by = 'songs'  )
#song_clu.to_pickle("data/digitization/clu_song_emb_200.pickle")

tag_gnr_title_clu = clu_km.clustering_test( merged_t.tag_gnr_title(id_title = id_title) , by = 'tag_gnr_title')
tag_gnr_title_clu.to_pickle("data/digitization/clu_tag_gnr_title_emb_100.pickle")
#
# C. singer
singer_clu = clu_km.clustering_test(merged_t.title_singer(id_title = id_title) , by = 'title_singer')
singer_clu.to_pickle("data/digitization/clu_singer_emb_100.pickle")


# D. tag : complex_
tag_clu = clu_km.clustering_test( by = 'tags'  )
tag_clu.to_pickle("data/digitization/clu_tag_emb_30.pickle")


# E. dtl_genre 원핫 인코딩을 했으므로 이 단계 생략


# F . album  아래에서 df 랑 clustering이랑 코드가 합쳐져있음.
# emb_df   = clu_km.clustering_test( by = 'album'  )
# emb_df.to_pickle('data/digitization/clu_album_emb100.pickle' )
#album_df.to_pickle('data/digitization/album_df_.pickle') # 데이타 프레임 윗단계



#%%

#
# # A. Song  : complex_
# song_clu = clu_km.clustering_song( clu = 200 )
# song_clu.to_pickle("data/digitization/clu_song_emb_200.pickle")
#
# # B. word
# tag_gnr_title_clu = clu_km.clustering_( merged_t.tag_gnr_title() , clu = 100)
# tag_gnr_title_clu.to_pickle("data/digitization/clu_tag_gnr_title_emb_100.pickle")
#
#
# # C. singer
# singer_clu = clu_km.clustering_( merged_t.title_singer() , clu = 100)
# singer_clu.to_pickle("data/digitization/clu_singer_emb_100.pickle")
#
# # D. tag : complex_
# tag_clu = clu_km.clustering_tag( clu = 30  )
# tag_clu.to_pickle("data/digitization/clu_tag_emb_30.pickle")
#
#
# # E. dtl_genre 원핫 인코딩을 했으므로 이 단계 생략
#
#
# # F . album  아래에서 df 랑 clustering이랑 코드가 합쳐져있음.
# emb_df , album_df  = clu_km.clustering_album( clu = 100 )
# emb_df.to_pickle('data/digitization/clu_album_emb100.pickle' )
# album_df.to_pickle('data/digitization/album_df_.pickle') # 데이타 프레임 윗단계
