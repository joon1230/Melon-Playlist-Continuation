

def stop_title(): # 한국어ㅇㅇ 불용어사전
    stop_words= open( "badwords.txt" , 'r', encoding='utf-8').read()
    stop_words=stop_words.split('\n')
    stop_words_lst=list(set(stop_words))

    return stop_words_lst


def re_sub(series: pd.Series) :#-> pd.Series:
    series = series.str.replace(pat=r'[ㄱ-ㅎ]', repl=r'', regex=True)  # ㅋ 제거용
    series = series.str.replace(pat=r'[^\w\s]', repl=r'', regex=True)  # 특수문자 제거
    series = series.str.replace(pat=r'[ ]{2,}', repl=r' ', regex=True)  # 공백 제거
    series = series.str.replace(pat=r'[\u3000]+', repl=r'', regex=True)  # u3000 제거
    return series


class Clustering:

    def __init__(self, complex_, meta):
        self.complex_ = complex_
        self.meta = meta


    def clustering_(self, result, clu=100):
        """
        result = id 와 매핑한 [] 을 한 데이터  최종 프레임
                ex, title_singer / tag_gnr_title / tag_title  위에 있는 것들
        """
        print("clustering -ing")
        result_all = result

        # word2vec 하기
        result_ = {}
        for i, res in enumerate(result_all.complex_col):
            result_[i] = [str(re) for re in res]

        w2v_model = Word2Vec(result_.values(), size=100, window=40, min_count=1, sg=0)

        v_complex = np.asarray(w2v_model.wv.index2word)
        complex_vec = w2v_model.wv.vectors

        col = [f'vector{i}' for i in range(1, complex_vec.shape[1] + 1)]
        emb_df = pd.DataFrame(complex_vec, columns=col)
        emb_df['tv_complex'] = v_complex

        # kmeans
        kmeans = KMeans(n_clusters=clu)
        kmeans.fit(complex_vec)
        emb_df['label'] = kmeans.labels_

        emb_df = emb_df[['tv_complex', 'label'] + [f'vector{i}' for i in range(1, 101)]]

        return emb_df

    def clustering_song( self, clu=200 ):

        print("clustering -ing")

        ply_song = self.complex_[["id", "songs"]]

        songs_ = {}
        for i, sgs in tqdm(enumerate(ply_song.songs)):
            if len(sgs) != 0:
                songs_[i] = [str(song) for song in sgs]

        # embedding
        w2v_model = Word2Vec(songs_.values(), size=100, window=50, min_count=1, sg=0)

        v_songs = np.asarray(w2v_model.wv.index2word)
        song_vec = w2v_model.wv.vectors

        col = [f'vector{i}' for i in range(1, song_vec.shape[1] + 1)]
        emb_df = pd.DataFrame(song_vec, columns=col)
        emb_df['song_id'] = v_songs
        emb_df = emb_df[['song_id'] + col]

        # kmeans
        kmeans = KMeans(n_clusters=clu)
        kmeans.fit(song_vec)

        emb_df['label'] = kmeans.labels_
        emb_df = emb_df[['song_id', 'label'] + col]

        return emb_df

    def clustering_tag( self, clu=30):

        print("clustering -ing")

        id_tag =  self.complex_[["id", "tags"]].sort_values(by="id")

        # id_tag_df 저장
        id_tag.to_pickle('tag_df.pickle')

        result_ = {}
        for i, res in enumerate(id_tag.tags):
            result_[i] = [str(re) for re in res]

        w2v_model = Word2Vec(result_.values(), size=100, window=8, min_count=1, sg=0)

        v_complex = np.asarray(w2v_model.wv.index2word)
        complex_vec = w2v_model.wv.vectors

        col = [f'vector{i}' for i in range(1, complex_vec.shape[1] + 1)]
        emb_df = pd.DataFrame(complex_vec, columns=col)
        emb_df['tv_complex'] = v_complex

        # kmeans
        kmeans = KMeans( n_clusters=clu )
        kmeans.fit(complex_vec)
        emb_df['label'] = kmeans.labels_

        emb_df = emb_df[['tv_complex', 'label'] + [f'vector{i}' for i in range(1, 101)]]

        emb_df.to_pickle('clustering_tag_emb_30.pickle')

        return emb_df

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

        emb_df = emb_df[['album_id', 'label'] + col]

        return emb_df, pd.DataFrame(list(zip(self.complex_.id, result_.values())), columns=['id', 'ablums'])
