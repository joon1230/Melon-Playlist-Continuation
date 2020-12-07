import pandas as pd
import KakaoArena.Get_df as gd

class Merged_table():
    def __init__(self, complex_ , genre, meta ):
        self.complex_ = complex_
        self.meta = meta
        self.genre = genre

    def tag_title( self ):
        print ("tag_title_merge")
        # tar , title 함수 다 넣기
        get_df = gd.Get_data( self.complex_, self.genre, self.meta )
        id_title = get_df.get_title_df()
        id_tag = self.complex_[["id", "tags"]].sort_values(by="id")

        # 두개를 merge 하기
        complex_all = id_tag.merge(id_title, how="outer")

        # 두개 다 합치는 코드
        target_col = ['tags', "tok_title"]

        a = []
        for v in complex_all[target_col].values:
            if type(v[1]) != list:  # 타입에 float 인 것이 있어서 이렇게 처리
                a.append([v[0] + v[1]])
            else:
                a.append([v[0] + v[1]])
        b = complex_all["id"]

        result = pd.concat([b, pd.DataFrame(a)], axis=1)
        result = result.rename(columns={0: "complex_col"})

        return result


    def tag_gnr_title( self ):
        print("tag_gnr_title_merge")
        # tar , gnr , title 함수 다 넣기
        get_df = gd.Get_data( self.complex_, self.genre, self.meta )
        id_gnr = get_df.get_gnr_df()
        id_title = get_df.get_title_df()
        id_tag = self.complex_[["id", "tags"]].sort_values(by="id")

        # 세개를 merge 하기
        complex_all = id_tag.merge(id_gnr, how="outer").merge(id_title, how="outer")

        # 세개 다 합치는 코드
        target_col = ['tags', 'gnr', "tok_title"]

        a = []
        for v in complex_all[target_col].values:
            if type(v[1]) != list:  # 타입에 float 인 것이 있어서 이렇게 처리
                a.append([v[0] + [] + v[2]])
            else:
                a.append([v[0] + v[1] + v[2]])
        b = complex_all["id"]

        result = pd.concat([b, pd.DataFrame(a)], axis=1)
        result = result.rename(columns={0: "complex_col"})

        return result


    def title_singer( self ):
        print("title_singer_merge")
        get_df = gd.Get_data( self.complex_, self.genre, self.meta )
        id_title = get_df.get_title_df()
        id_singer = get_df.get_singer_df()

        # 두개를 merge 하기
        complex_all = id_singer.merge(id_title, how="outer")

        # 두개 다 합치는 코드
        target_col = ['singer', 'tok_title']

        a = []
        for v in complex_all[target_col].values:
            if type(v[1]) != list:  # 타입에 float 인 것이 있어서 이렇게 처리
                a.append([v[0] + v[1]])
            else:
                a.append([v[0] + v[1]])
        b = complex_all["id"]

        result = pd.concat([b, pd.DataFrame(a)], axis=1)
        result = result.rename(columns={0: "complex_col"})

        return result