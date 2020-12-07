import pandas as pd
from tqdm import tqdm


class Get_candidate:

    def __init__(self, train, tag_title_df ):
        self.train = train
        self.tag_title_df = tag_title_df

    def get_candidate( self, sim_dict, val, by='set'):
        """ 후보군 뽑기
        # val_songs , val
        # val_tags , ply_tagtitle , val
        # sim --> train + ply_tagtitle
        # by --> set , count
        """
        # global song_date

        val_songs = val.songs.values
        val_tags = val.tags.values
        #     val_date = pd.to_datetime(val.updt_date).dt.date

        sim_ply = pd.merge(self.train[['id', 'songs']], self.tag_title_df, on='id', how='left')
        res = {}
        i = 0
        for k in tqdm(sim_dict.keys()):
            sim = sim_ply.loc[sim_ply.id.isin(sim_dict[k])]
            if by == 'set':
                cand_songs = list(set([s for sgs in sim.songs for s in sgs if not (s in val_songs[i])]))
            elif by == 'count':
                cand_songs = [s for sgs in sim.songs for s in sgs if not (s in val_songs[i])]
            # and (song_date[s] < val_date[i])]
            cand_tags = [t for tgs in sim.complex_col for t in tgs if not (t in val_tags[i])]
            res[k] = {'songs': cand_songs, 'tags': cand_tags}
            i += 1
        return res