import datetime
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import pickle

class Select_candidate:

    def __init__(self):
        pass


    # 후보군 선별하기 : 유사한 플레이리스트를 찾는 함수와 그 플레이리스트 노래. 태그 모으기
    def get_sim_ply(self , val_ply, train_ply, top_ply=30):  # 유사도 높은 playlist뽑기
        """
        val_ply :  df - test_ply :  비교 대상이 test일 경우, test_ply 을 넣기
            예측할 playlist ( 노래기반 val_song , word 기반 val_word , 둘다 val_ply )
        train_ply : df
            비교대상 playlist ( 노래기반 train_song , word 기반 train_word , 둘다 train_ply )
        top_ply : int
            상위 노래 갯수
        """

        n_train = len(train_ply)
        n_val = len(val_ply)
        train_key = dict(zip(range(n_train), train_ply.id))
        val_key = dict(zip(range(n_val), val_ply.id))

        sim_matrix = cosine_similarity(val_ply.set_index('id'), train_ply.set_index('id'))

        sim_ply_dict = {}
        for i, val_ in tqdm(enumerate(sim_matrix)):
            sim_ply_dict[val_key[i]] = [train_key[p] for p in np.argsort(-val_)[:top_ply]]

        return sim_ply_dict


    # 분산처리 아닌 분산처리 - 너무 많이 하면 에러나서 이렇게 처리를 해줌. 그냥 이것만 하면 됨.
    def dist_get_sim_ply( self , start, inter , val_ply  , train_ply  , tn = 30 ):
        res = {}
        max_n = len(val_ply)
        for i in range( start , max_n ,  inter ):
            try:
                tmp = self.get_sim_ply( val_ply = val_ply.iloc[i : i + inter] ,train_ply = train_ply , top_ply = tn )
                res.update( tmp )
                print(f'{i+inter} 만큼 했다')
            except:
                print( f'ERROR { i } 에서 Save')
                return res
        return res

    # 피클로 저장하기
    def save_to_pickle( self, ob , f_path ):
        with open( f_path , 'wb') as f:
            pickle.dump( ob , f)

    # 피클로 불러오기
    def load_pickle( self , f_path):
        with open(f_path, 'rb') as f:
            return pickle.load(f)
