from sm.engine.extra_feats.formula_imager_segm import compute_sf_images
from sm.engine.extra_feats.formula_img_validator import sf_image_metrics, sf_image_metrics_est_fdr
from sm.engine.search_algorithm import SearchAlgorithm
from sm.engine.util import logger


class MSMExtraFeats(SearchAlgorithm):

    def __init__(self, sc, ds, formulas, fdr, ds_config):
        super(MSMExtraFeats, self).__init__(sc, ds, formulas, fdr, ds_config)
        self.metrics = ['chaos', 'spatial', 'spectral', 'image_corr_01', 'image_corr_02', 'image_corr_03',
                        'image_corr_12', 'image_corr_13', 'image_corr_23', 'snr', 'nnz_percent', 'peak_int_diff_0',
                        'peak_int_diff_1', 'peak_int_diff_2', 'peak_int_diff_3', 'quart_1',
                        'quart_2', 'quart_3', 'ratio_peak_01', 'ratio_peak_02', 'ratio_peak_03', 'ratio_peak_12',
                        'ratio_peak_13', 'ratio_peak_23', 'percentile_10', 'percentile_20', 'percentile_30',
                        'percentile_40', 'percentile_50', 'percentile_60', 'percentile_70', 'percentile_80',
                        'percentile_90']
        self.max_fdr = 0.5

    def search(self):
        logger.info('Running molecule search')
        sf_images = compute_sf_images(self.sc, self.ds, self.formulas.get_sf_peak_df(),
                                      self.ds_config['image_generation']['ppm'])
        all_sf_metrics_df = self.calc_metrics(sf_images)
        sf_metrics_fdr_df = self.estimate_fdr(all_sf_metrics_df)
        sf_metrics_fdr_df = self.filter_sf_metrics(sf_metrics_fdr_df)
        return sf_metrics_fdr_df, self.filter_sf_images(sf_images, sf_metrics_fdr_df)

    def calc_metrics(self, sf_images):
        all_sf_metrics_df = sf_image_metrics(sf_images, self.sc, self.formulas, self.ds, self.ds_config)
        return all_sf_metrics_df

    def estimate_fdr(self, all_sf_metrics_df):
        sf_metrics_fdr_df = sf_image_metrics_est_fdr(all_sf_metrics_df, self.formulas, self.fdr)
        return sf_metrics_fdr_df

    def filter_sf_metrics(self, sf_metrics_df):
        return sf_metrics_df#[sf_metrics_df.fdr <= self.max_fdr]
