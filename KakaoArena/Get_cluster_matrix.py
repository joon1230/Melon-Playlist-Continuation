from collections import Counter
import numpy as np
import scipy.sparse as spr
import pandas as pd

class Get_cluster_matrix:

    def __init__(self, complex_,  meta , genre ):
        self.complex_ = complex_
        self.meta = meta
        self.genre = genre


    def get_cluster_matrix( self , df, cls_data, by='song'):
        """
        # df = dataframe
                play list  / tag_title_df ...
        # cls_dat = dataframe
                clustered data
        # by = song, word, singer, tag, genre, album 중에서 고르기
        """

        df = df
        cls_data = cls_data

        n_cluster = len(cls_data.label.unique())
        n_plylst = len(df)
        cls_data = cls_data[cls_data.columns[:2]]

        cl_val = dict(zip(cls_data.iloc[:, 0], cls_data.iloc[:, 1]))

        if by == 'song':   # df  = complex_ , cls_data = song_clustering
            df = df[['id', 'songs']]
            cl_ply_values = [dict(Counter([cl_val.get(str(v)) for v in vs])) for vs in df.songs]

        elif by == 'word':
            df = df[['id', 'complex_col']]
            cl_ply_values = [dict(Counter([cl_val.get(str(v)) for v in vs if cl_val.get(str(v)) != None])) for vs in
                             df.complex_col]

        elif by == 'singer':
            df = df[['id', 'complex_col']]
            cl_ply_values = [dict(Counter([cl_val.get(str(v)) for v in vs if cl_val.get(str(v)) != None])) for vs in
                             df.complex_col]

        elif by == 'tag':
            df = df[['id', 'tags']].reset_index(drop=True)
            cl_ply_values = [dict(Counter([cl_val.get(str(v)) for v in vs if cl_val.get(str(v)) != None])) for vs in
                             df.tags]

        elif by == 'genre':
            df = df[['id', 'gnr']]
            cl_ply_values = [dict(Counter([cl_val.get(str(v)) for v in vs if cl_val.get(str(v)) != None])) for vs in df.gnr]

        elif by == 'album':
            df = df[['id', 'albums']]
            cl_ply_values = [dict(Counter([cl_val.get(str(v)) for v in vs if cl_val.get(str(v)) != None])) for vs in
                             df.albums]

        col_dot = [list(d.keys()) for d in cl_ply_values]
        row = np.repeat(range(n_plylst), [len(c) for c in col_dot])
        col = [c for cols in col_dot for c in cols]
        data = [d for k in cl_ply_values for d in list(k.values())]
        matrix_cluster = spr.csr_matrix((data, (row, col)), shape=(n_plylst, n_cluster))

        return pd.concat([df.id, pd.DataFrame(matrix_cluster.toarray(), columns=[f'{by}_cl{i}' for i in range(n_cluster)])],
                         axis=1)


    def get_genre_matrix( self ):

        # 노래와 장르 dictionary 함
        d_g = self.meta.song_gn_dtl_gnr_basket.values
        g = self.meta.song_gn_gnr_basket.values

        song_gnr = dict(zip( self.meta.id, [d_g[i] + g[i] for i in range(len(d_g))]))
        gnrs = [[[g for s in sgs for g in song_gnr[s]]] for sgs in self.complex_.songs.values]
        gnrs = pd.DataFrame(gnrs, columns=['gnr'], index= self.complex_.id)

        gnr_new = pd.DataFrame(list(zip( self.genre.index, range(len( self.genre.index)))), columns=['id', 'label'])

        res = self.get_cluster_matrix(gnrs.reset_index(), gnr_new, by='genre')
        res.columns = ['id'] + list( self.genre.values)

        return res


