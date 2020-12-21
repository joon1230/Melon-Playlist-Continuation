import KakaoArena.Select_candidate as sc
import KakaoArena.Get_candidate as gc

candidate = sc.Select_candidate()


get_songtag = gc.Get_candidate(train, tag_title_df)


s_can = get_songtag.get_candidate( s_sim , val, by ="set" )
s_can = fdate.filter_date( s_can )

w_can = get_songtag.get_candidate( w_sim , val, by ="set" )
w_can = fdate.filter_date( w_can  )

sin_can = get_songtag.get_candidate( sin_sim , val, by ="set" )
sin_can = fdate.filter_date( sin_can)

t_can = get_songtag.get_candidate( t_sim , val, by ="set")
t_can = fdate.filter_date( t_can )

g_can = get_songtag.get_candidate( g_sim , val, by ="set")
g_can = fdate.filter_date( g_can )

ply_can = get_songtag.get_candidate( ply_sim , val, by ="set" )
ply_can = fdate.filter_date( ply_can)