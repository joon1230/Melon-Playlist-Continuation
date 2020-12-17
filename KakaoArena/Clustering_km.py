from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from tqdm import tqdm




class Clustering:

    def __init__(self, complex_, meta):
        self.complex_ = complex_
        self.meta = meta
        self.config = {"songs" : { "size":100 , "window" : 50, "min_count":1, "sg":0 , "clu" : 200},
                       "tags" : { 'size' : 100 , "window" : 8, 'min_count' : 1 , "sg": 0 , "clu": 30},
                       "album" : { 'size' : 100 , "window" : 8, 'min_count' : 1 , "sg": 0 , "clu": 100},
                       "title_singer" : { 'size' : 100 , "window" : 40, 'min_count' : 1 , "sg": 0, "clu": 100},
                       "tag_gnr_title" : { 'size' : 100 , "window" : 40, 'min_count' : 1 , "sg": 0 , "clu": 100},
                       "tag_title" : { 'size' : 100 , "window" : 40, 'min_count' : 1 , "sg": 0 , "clu": 100}}



    def clustering_album( self , clu = 100):

        print("clustering -ing")

        song_album = dict(zip(self.meta.id, self.meta.album_id))

        result_ = {}
        for i, res in enumerate(self.complex_.songs):
            result_[i] = list(set([str(song_album[re]) for re in res]))

        w2v_model = Word2Vec(result_.values(), size=100, window=8, min_count=1, sg=0)

        v_complex = np.asarray(w2v_model.wv.index2word)
        complex_vec = w2v_model.wv.vectors

        col = [f'vector{i}' for i in range(1, complex_vec.shape[1] + 1)]
        emb_df = pd.DataFrame(complex_vec, columns=col)
        emb_df['album_id'] = v_complex

        # kmeans
        kmeans = KMeans(n_clusters= clu)
        kmeans.fit(complex_vec)
        emb_df['label'] = kmeans.labels_

        emb_df = emb_df[['album_id', 'label']]

        return emb_df, pd.DataFrame(list(zip(self.complex_.id, result_.values())), columns=['id', 'ablums'])


    def clustering_test( self, df = None, by = 'song'):
        """
        title_singer / tag_gnr_title / tag_title --> 별도의 데이터 프레임 필요 ( 외부변수 받음 )
        song / tag / 외부변수 받지 않음
        ## tag는 변수에 사용x
        """
        self.clu = self.config[by].pop("clu")
        is_multi = by in ["title_singer","tag_gnr_title"]

        if is_multi:
            df.columns = ["id", by]
        elif by == 'album':
            song_album = dict(zip(self.meta.id, self.meta.album_id))
        else:
            df = self.complex_[["id", by]].sort_values(by="id")

        print(f"digitize ## {by} ")
        # if by == "tag":
        #     df.to_pickle('tag_df.pickle')
        items = {}
        if by == 'album' :
            for i, itms in enumerate(self.complex_.songs):
                items[i] = list(set([str(song_album[itm]) for itm in itms]))
        else:
            for i, itms in enumerate(df[by]):
                items[i] = [str(itm) for itm in itms]

        emb_df = self.embedding(items, by)
        return emb_df


    def embedding(self,items,by):
        clu = self.clu
        cfg = self.config[by]
        w2v_model = Word2Vec(items.values(), **cfg )

        idx_items = np.asarray(w2v_model.wv.index2word)
        items_vec = w2v_model.wv.vectors

        print( "%-20s"%('embedding...') , end = '\r' )
        col = [f'vector{i}' for i in range(1, items_vec.shape[1] + 1)]
        emb_df = pd.DataFrame(items_vec, columns=col)
        emb_df[f'{by}_id'] = idx_items
        emb_df = emb_df[[f'{by}_id'] + col]


        # kmeans
        print("%-20s"%('clustering...'), end='\r')
        kmeans = KMeans(n_clusters=clu)
        kmeans.fit(items_vec)

        emb_df['label'] = kmeans.labels_
        emb_df = emb_df[[f'{by}_id', 'label']]
        print(f'completed : {by}' , end = '\n\n')
        return emb_df




