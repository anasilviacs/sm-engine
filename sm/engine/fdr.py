from cStringIO import StringIO

import numpy as np
import pandas as pd

from sm.engine.util import logger

DECOY_ADDUCTS = ['+He', '+Li', '+Be', '+B', '+C', '+N', '+O', '+F', '+Ne', '+Mg', '+Al', '+Si', '+P', '+S', '+Cl', '+Ar', '+Ca', '+Sc', '+Ti', '+V', '+Cr', '+Mn', '+Fe', '+Co', '+Ni', '+Cu', '+Zn', '+Ga', '+Ge', '+As', '+Se', '+Br', '+Kr', '+Rb', '+Sr', '+Y', '+Zr', '+Nb', '+Mo', '+Ru', '+Rh', '+Pd', '+Ag', '+Cd', '+In', '+Sn', '+Sb', '+Te', '+I', '+Xe', '+Cs', '+Ba', '+La', '+Ce', '+Pr', '+Nd', '+Sm', '+Eu', '+Gd', '+Tb', '+Dy', '+Ho', '+Ir', '+Th', '+Pt', '+Os', '+Yb', '+Lu', '+Bi', '+Pb', '+Re', '+Tl', '+Tm', '+U', '+W', '+Au', '+Er', '+Hf', '+Hg', '+Ta']
SF_LIST_SEL = ('SELECT af.id '
               'FROM agg_formula af '
               'JOIN formula_db db ON db.id = af.db_id '
               'WHERE db.id = %s')


class FDR(object):

    def __init__(self, job_id, db_id, decoy_sample_size, target_adducts, db):
        self.job_id = job_id
        self.db_id = db_id
        self.decoy_sample_size = decoy_sample_size
        self.db = db
        self.target_adducts = target_adducts
        self.td_df = None
        self.fdr_levels = [0.01, 0.05, 0.1, 0.2, 0.5]

    @staticmethod
    def _decoy_adduct_gen(sf_ids, target_adducts, decoy_adducts_cand, decoy_sample_size):
        for sf_id in sf_ids:
            for ta in target_adducts:
                for da in np.random.choice(decoy_adducts_cand, size=decoy_sample_size, replace=False):
                    yield (sf_id, ta, da)

    def _save_target_decoy_df(self):
        buf = StringIO()
        df = self.td_df.copy()
        df.insert(0, 'db_id', self.db_id)
        df.insert(0, 'job_id', self.job_id)
        df.to_csv(buf, index=False, header=False)
        buf.seek(0)
        self.db.copy(buf, 'target_decoy_add', sep=',')

    def decoy_adduct_selection(self):
        sf_ids = [r[0] for r in self.db.select(SF_LIST_SEL, self.db_id)]
        decoy_adduct_cand = list(set(DECOY_ADDUCTS) - set(self.target_adducts))
        self.td_df = pd.DataFrame(self._decoy_adduct_gen(sf_ids, self.target_adducts,
                                                         decoy_adduct_cand, self.decoy_sample_size),
                                  columns=['sf_id', 'ta', 'da'])
        self._save_target_decoy_df()

    @staticmethod
    def _msm_fdr_map(target_msm, decoy_msm):
        target_msm_hits = pd.Series(target_msm.msm.value_counts(), name='target')
        decoy_msm_hits = pd.Series(decoy_msm.msm.value_counts(), name='decoy')
        msm_df = pd.concat([target_msm_hits, decoy_msm_hits], axis=1).fillna(0).sort_index(ascending=False)
        msm_df['target_cum'] = msm_df.target.cumsum()
        msm_df['decoy_cum'] = msm_df.decoy.cumsum()
        msm_df['fdr'] = msm_df.decoy_cum / msm_df.target_cum
        return msm_df.fdr

    def _digitize_fdr(self, fdr_df):
        df = fdr_df.copy().sort_values(by='msm', ascending=False)
        msm_levels = [df[df.fdr < fdr_thr].msm.min() for fdr_thr in self.fdr_levels]
        df['fdr_d'] = 1.
        for msm_thr, fdr_thr in zip(msm_levels, self.fdr_levels):
            row_mask = np.isclose(df.fdr_d, 1.) & np.greater_equal(df.msm, msm_thr)
            df.loc[row_mask, 'fdr_d'] = fdr_thr
        df['fdr'] = df.fdr_d
        return df.drop('fdr_d', axis=1)

    def estimate_fdr(self, msm_df):
        logger.info('Estimating FDR...')

        target_fdr_df_list = []
        for ta in self.target_adducts:
            target_msm = msm_df.loc(axis=0)[:,ta]

            msm_fdr_list = []
            for i in range(self.decoy_sample_size):
                sf_da_list = map(tuple, self.td_df[self.td_df.ta == ta][['sf_id', 'da']][i::self.decoy_sample_size].values)
                decoy_msm = msm_df.loc[sf_da_list]
                msm_fdr = self._msm_fdr_map(target_msm, decoy_msm)
                msm_fdr_list.append(msm_fdr)

            msm_fdr_avg = pd.Series(pd.concat(msm_fdr_list, axis=1).median(axis=1), name='fdr')
            target_fdr = self._digitize_fdr(target_msm.join(msm_fdr_avg, on='msm'))
            target_fdr_df_list.append(target_fdr.drop('msm', axis=1))

        return pd.concat(target_fdr_df_list, axis=0)
