#!/usr/bin/env python
# coding: utf-8

import pandas as pd

"""
stop_title : 한국어 불용어 사전
re_sub : 타이틀 전처리
"""

def stop_title(): # 한국어ㅇㅇ 불용어사전
    stop_words= open( "KakaoArena/badwords.txt" , 'r', encoding='utf-8').read()
    stop_words=stop_words.split('\n')
    stop_words_lst=list(set(stop_words))

    return stop_words_lst


def re_sub(series: pd.Series) :#-> pd.Series:
    series = series.str.replace(pat=r'[ㄱ-ㅎ]', repl=r'', regex=True)  # ㅋ 제거용
    series = series.str.replace(pat=r'[^\w\s]', repl=r'', regex=True)  # 특수문자 제거
    series = series.str.replace(pat=r'[ ]{2,}', repl=r' ', regex=True)  # 공백 제거
    series = series.str.replace(pat=r'[\u3000]+', repl=r'', regex=True)  # u3000 제거
    return series

