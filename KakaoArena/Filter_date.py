import pandas as pd
import numpy as np
import numpy.linalg
from tqdm import tqdm
import datetime
import pickle


class Filter_date:  # meta date 를 filter 해주는 class

    def __init__(self , complex_ , meta) :
        self.complex_  = complex_
        self.meta = meta

    ##  1. 이상한 date 를 전처리 해줘서 meta data 에 넣기!

    def update_date( self ):

        # 노래 매핑
        map_df = self.complex_.loc[:, ['songs', 'updt_date']]

        ## time_delay..
        map_unnest = np.dstack((np.concatenate(map_df.songs.values), np.repeat(map_df.updt_date.values, list(map(len, map_df.songs)))))

        # 데이터 프레임으로 변환 후,
        map_df = pd.DataFrame(data=map_unnest[0], columns=map_df.columns)


        map_df['updt_date'] = [int(v[:10].replace("-", '')) for v in map_df.updt_date.values]
        date_df = map_df.groupby(['songs'])['updt_date'].agg({'max'}).reset_index().rename(columns={'max': 'updt_date'})
        date_df['songs'] = date_df['songs'].astype(int)

        # date_df 와 meta 속 song의 발행 날짜 merge
        meta_song = self.meta[['id', 'issue_date']].rename(columns={'id': 'songs'})
        date_df = date_df.merge(meta_song, how='outer', on='songs')

        #  meta 데이터의 노래더 느리면 노래를 플레이리스트 데이터 기준으로 바꿈
        date_df.loc[date_df['updt_date'] < date_df['issue_date'], 'issue_date'] = date_df.loc[date_df['updt_date'] < date_df['issue_date'], 'updt_date']

        # 메타데이터 변경
        self.meta["issue_date"] = date_df['issue_date'].astype(int)

        return self.meta

    # 윤소가 짠 date '0 '인 것을 처리 - 위의 update_date는 안에 포함되어있음
    def kill_zeros( self ):
        # 1차 방역

        meta = self.update_date()
        zeroloc = []
        for i in range(len(meta)):
            if meta.iloc[i, 1] == 0:
                zeroloc.append(i)
        for i in zeroloc:
            x = meta.loc[i].album_id
            y = meta[meta['album_id'] == x]
            med_date = y.issue_date.quantile(q=0.5, interpolation='nearest')
            meta.issue_date.loc[i] = med_date

            # 2차 방역
        zeroloc2 = []
        for i in range(len(meta)):
            if meta.iloc[i, 1] == 0:
                zeroloc2.append(i)
        for i in zeroloc2:
            x = meta.loc[i].album_id
            y = meta[meta['album_id'] == x]
            med_date = y.issue_date.max()
            meta.issue_date.loc[i] = med_date

            # 3차 방역
        zeroloc3 = []
        for i in range(len(meta)):
            if meta.iloc[i, 1] == 0:
                zeroloc3.append(i)
        songs = list(set(meta[meta['issue_date'] == 0].song_name))
        for i in zeroloc3:
            x = meta[meta['song_name'].isin(songs)]
            med_date = x.issue_date.quantile(q=0.5, interpolation='nearest')
            meta.issue_date.loc[i] = med_date
        year = []
        for i in range(len(meta)):
            x = str(meta.issue_date.loc[i])[:4]
            year.append(int(x))

        meta['issue_year'] = year

        with open('data/meta_final.pickle', 'wb') as f:
            pickle.dump(meta, f)

        return meta

    ########################################################

    # 사전에 날짜 에러 처리 하는 함수
    def get_song_date_dict( self ): # song_meta를 받아서 날짜와 매핑한다.
        # 간혹 20050200 같은 놈들은 20050101
        # 0 인놈들은 1990 으로 변경
        a = []
        for t in self.meta.issue_date:
            try:
                a.append(datetime.datetime.strptime( str(t) , '%Y%m%d' ).date())
            except:
                try:
                    a.append(datetime.datetime.strptime( str(t)[:4] , '%Y' ).date())
                except:
                    a.append(datetime.datetime.strptime( '1990' , '%Y' ).date())
        return dict(zip( self.meta.id , a))


    def filter_date( self, candi ):

        song_date = self.get_song_date_dict()
        global val
        val_date = pd.to_datetime(val.updt_date).dt.date
        j = 0
        for i in tqdm(candi):
            candi[i]['songs'] = [ s for s in candi[i]['songs'] if song_date[s] <= val_date[j]]
            j += 1
        return candi