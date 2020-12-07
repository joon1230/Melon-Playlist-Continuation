from tqdm import tqdm
import itertools
import re
import pandas as pd
import sentencepiece as spm

import KakaoArena.Pre_title as pre

class Get_data:

    def __init__(self, complex_  , genre, meta ) :
        self.complex_ = complex_
        self.genre = genre
        self.meta = meta

    # / 를 , 로 바꿔주는 함수
    def change_(self, x):
        x = [x]
        if "/" in x[0]:
            a = x[0].split("/")
            return [a[0], a[1]]
        else:
            return x


    # 장르데이터를 전처리 하고, 딕셔너리 형태로 바꿔줌
    def gnr_to_dic( self ):
        # 대분류 장르만 뽑은 후, 두개 장르를 나누기
        genre = pd.DataFrame(self.genre, columns=["gnr_name"]).reset_index().rename(columns={'index': "gnr_code"})
        gnr_code = genre[genre.iloc[:, 0].str[-2:] == '00']
        gnr_code["gnr_name"] = gnr_code["gnr_name"].map( self.change_ )

        # 장르 추가  : GN0000(장르없음), GN9000(장르알수없음)
        gnr_code = gnr_code.append(pd.DataFrame([['GN9000', None], ['GN0000', None]],
                                                columns=["gnr_code", "gnr_name"])).sort_values('gnr_code').reset_index(drop=True)

        # 장르코드 : 장르명의 딕셔너리 생성
        dict_gnr = gnr_code.set_index('gnr_code')['gnr_name'].to_dict()

        return dict_gnr


    def get_gnr_df( self ):  # 플레이리스트에 담긴 모오오오든 장르 가져오기

        df = self.complex_
        song_dict = dict(zip(self.meta.id.values, self.meta.song_gn_gnr_basket.values))
        gnr_dict = self.gnr_to_dic()

        df = df[['id', 'songs']]
        res = []
        for i in tqdm(df.values):
            if i[1] == []:
                res.append([i[0], []])
            else:
                res.append([i[0], list(set(
                    [g for s in i[1] for gc in song_dict.get(s) if gnr_dict.get(gc) != None for g in gnr_dict.get(gc)]))])
        return pd.DataFrame(res, columns=['id', 'gnr'])




    # 타이틀을 토크나이즈 하는 함수 ** 고생한 윤소의 작품 **
    def js_title(  self ):

        self.complex_['plylst_title'] = pre.re_sub(self.complex_['plylst_title'])

        title = self.complex_["plylst_title"]
        title.to_csv('title.txt', index=False, header=None, sep='\n')
        # plylst_title = pd.read_csv('title.txt', sep='\n', header=None)
        # header=None안써주면 0번째 제목이 컬럼으로 감

        input_file = 'title.txt'  # 아까 만들어준 traindata의 텍스트파일
        vocab_size = 32000
        model_name = 'subword_tokenizer_kor'  # 모델이름 (맘대로 커스텀가능)
        model_type = 'bpe'
        user_defined_symbols = '[PAD],[UNK],[CLS],[SEP],[MASK],[UNK1],[UNK2],[UNK3],[UNK4]'  # 내가 커스텀해주는거네

        input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --user_defined_symbols=%s --model_type=%s'
        cmd = input_argument % (input_file, model_name, vocab_size, user_defined_symbols, model_type)

        # cmd를 Train시켜준다.
        spm.SentencePieceTrainer.Train(cmd)

        # 생성된 model파일 불러오기
        vocab_file = 'subword_tokenizer_kor.model'
        vocab = spm.SentencePieceProcessor()
        vocab.load(vocab_file)

        jj = []
        jump_tok = []
        for i in range(len(self.complex_)):
            t = self.complex_['plylst_title'].iloc[i]
            tt = t.split()
            sentence = []
            for n in range(len(tt)):
                sent = vocab.encode_as_pieces(tt[n])
                sentence.append(sent)
            jj.append(list(itertools.chain(*sentence)))

        for x in jj:
            ss = []
            for h in x:
                a = re.sub('▁', '', h)  # '▁' 문자 제거.
                ss.append(a)
            jump_tok.append(ss)

        # 전처리(조사 빼기)
        js_lst = []
        for i in range(len(jump_tok)):
            res = jump_tok[i]
            jump = []
            for x in range(len(res)):
                if re.search('에$', res[x]):
                    l = re.sub(r'에$', r'', res[x])
                    jump.append(l)
                elif re.search('에게$', res[x]):
                    l = re.sub(r'에게$', r'', res[x])
                    jump.append(l)
                elif re.search('의$', res[x]):
                    l = re.sub(r'의$', r'', res[x])
                    jump.append(l)
                elif re.search('가$', res[x]):
                    l = re.sub(r'가$', r'', res[x])
                    jump.append(l)
                elif re.search('이$', res[x]):
                    l = re.sub(r'이$', r'', res[x])
                    jump.append(l)
                elif re.search('을$', res[x]):
                    l = re.sub(r'을$', r'', res[x])
                    jump.append(l)
                elif re.search('를$', res[x]):
                    l = re.sub(r'를$', r'', res[x])
                    jump.append(l)
                elif re.search('하게$', res[x]):
                    l = re.sub(r'하게$', r'', res[x])
                    jump.append(l)
                elif re.search('에서$', res[x]):
                    l = re.sub(r'에서$', r'', res[x])
                    jump.append(l)
                elif re.search('들$', res[x]):
                    l = re.sub(r'들$', r'', res[x])
                    jump.append(l)
                elif re.search('과$', res[x]):
                    l = re.sub(r'과$', r'', res[x])
                    jump.append(l)
                elif re.search('와$', res[x]):
                    l = re.sub(r'와$', r'', res[x])
                    jump.append(l)
                else:
                    jump.append(res[x])
            js_lst.append(jump)

        jumpsents = []
        for x in range(len(js_lst)):
            for i in js_lst[x]:
                filt = [h for h in js_lst[x] if len(h) > 1]
            jumpsents.append(filt)
        return jumpsents


    def get_title_df( self ):

        stop_words_lst = pre.stop_title()
        juump = self.js_title()

        sent_lst = []
        for w in juump:
            ress = []
            for i in range(len(w)):
                if w[i] not in stop_words_lst:
                    ress.append(w[i])
            sent_lst.append(ress)

        # 토크나이즈 한것을 넣기
        self.complex_['tok_title'] = sent_lst
        plyid_title = self.complex_[["id", "tok_title"]]
        plyid_title = plyid_title.sort_values(by="id")

        return plyid_title

    # 노래와 가수

    def get_singer_df( self ):
        singer_meta = self.meta[["id", "artist_name_basket"]]
        singer_dic = dict(zip(singer_meta.id.values, singer_meta.artist_name_basket.values))


        t = self.complex_[['id', 'songs']]
        ply_singer = {}
        for vec in t.values:
            songs = vec[1]
            ply_singer[vec[0]] = [
                list(set(gn for song in songs if singer_dic.get(song) != None for gn in singer_dic[song]))]

        # 딕셔너리를 데이터프레임으로 만들기
        ply_id = pd.DataFrame(ply_singer.keys())
        singer = pd.DataFrame(ply_singer.values())

        id_singer = pd.concat([ply_id, singer], axis=1)
        id_singer.columns = ["id", "singer"]
        id_singer = id_singer.sort_values(by="id")

        return id_singer
