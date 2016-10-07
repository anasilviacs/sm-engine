"""
Classes and functions for isotope image validation
"""
import numpy as np
import pandas as pd
from operator import mul, add

from pyImagingMSpec.image_measures import isotope_image_correlation, isotope_pattern_match
# from pyImagingMSpec.image_measures import measure_of_chaos
from cpyImagingMSpec import measure_of_chaos
from new_features import isotope_image_correlation_sd, snr_img, percent_nnz, spectra_int_diff, quartile_pxl, decile_pxl, \
    ratio_peaks
from pyImagingMSpec import smoothing


class ImgMeasures(object):
    """ Container for isotope image metrics

    Args
    ----------
    chaos : float
        measure of chaos, pyImagingMSpec.image_measures.measure_of_chaos
    image_corr : float
        isotope image correlation, pyImagingMSpec.image_measures.isotope_image_correlation
    pattern_match : float
        theoretical pattern match, pyImagingMSpec.image_measures.isotope_pattern_match
    image_corr_01/02/03/12/13/23 : float
        correlations between image pairs (rather than average)
    snr: float
        signal to noise ratio
    nnz_percent: float
        percent of non-zero values
    peak_int_diff_0/.../4:
        difference between the measured and the theoretical intensity of the first 5 isotopic peaks
    quart_1/2/3:
        number of pixels in each quartile
    ratio_peak_01/02/03/12/13/23:
        ratio between pairs of peaks
    percentile_10/20/.../90:
        number of pixels in 10th percentiles
    """

    def __init__(self, chaos, image_corr, pattern_match, image_corr_01, image_corr_02, image_corr_03, image_corr_12, image_corr_13,
                 image_corr_23, snr, nnz_percent, peak_int_diff_0, peak_int_diff_1, peak_int_diff_2,
                 peak_int_diff_3, quart_1, quart_2, quart_3, ratio_peak_01, ratio_peak_02,
                 ratio_peak_03, ratio_peak_12, ratio_peak_13, ratio_peak_23, percentile_10, percentile_20,
                 percentile_30, percentile_40, percentile_50, percentile_60, percentile_70, percentile_80,
                 percentile_90):

        self.chaos = chaos
        self.image_corr = image_corr
        self.pattern_match = pattern_match
        self.image_corr_01 = image_corr_01
        self.image_corr_02 = image_corr_02
        self.image_corr_03 = image_corr_03
        self.image_corr_12 = image_corr_12
        self.image_corr_13 = image_corr_13
        self.image_corr_23 = image_corr_23
        self.snr = snr
        self.nnz_percent = nnz_percent
        self.peak_int_diff_0 = peak_int_diff_0
        self.peak_int_diff_1 = peak_int_diff_1
        self.peak_int_diff_2 = peak_int_diff_2
        self.peak_int_diff_3 = peak_int_diff_3
        self.quart_1 = quart_1
        self.quart_2 = quart_2
        self.quart_3 = quart_3
        self.ratio_peak_01 = ratio_peak_01
        self.ratio_peak_02 = ratio_peak_02
        self.ratio_peak_03 = ratio_peak_03
        self.ratio_peak_12 = ratio_peak_12
        self.ratio_peak_13 = ratio_peak_13
        self.ratio_peak_23 = ratio_peak_23
        self.percentile_10 = percentile_10
        self.percentile_20 = percentile_20
        self.percentile_30 = percentile_30
        self.percentile_40 = percentile_40
        self.percentile_50 = percentile_50
        self.percentile_60 = percentile_60
        self.percentile_70 = percentile_70
        self.percentile_80 = percentile_80
        self.percentile_90 = percentile_90

    @staticmethod
    def _replace_nan(v, new_v=0):
        if not v or np.isinf(v) or np.isnan(v):
            return new_v
        else:
            return v

    def to_tuple(self, replace_nan=True):
        """ Convert metrics to a tuple

        Args
        ------------
        replace_nan : bool
            replace invalid metric values with the default one
        Returns
        ------------
        : tuple
            tuple of metrics
        """
        if replace_nan:
            return (self._replace_nan(self.chaos),
                    self._replace_nan(self.image_corr),
                    self._replace_nan(self.pattern_match),
                    self._replace_nan(self.image_corr_01),
                    self._replace_nan(self.image_corr_02),
                    self._replace_nan(self.image_corr_03),
                    self._replace_nan(self.image_corr_12),
                    self._replace_nan(self.image_corr_13),
                    self._replace_nan(self.image_corr_23),
                    self._replace_nan(self.snr),
                    self._replace_nan(self.nnz_percent),
                    self._replace_nan(self.peak_int_diff_0),
                    self._replace_nan(self.peak_int_diff_1),
                    self._replace_nan(self.peak_int_diff_2),
                    self._replace_nan(self.peak_int_diff_3),
                    self._replace_nan(self.quart_1),
                    self._replace_nan(self.quart_2),
                    self._replace_nan(self.quart_3),
                    self._replace_nan(self.ratio_peak_01),
                    self._replace_nan(self.ratio_peak_02),
                    self._replace_nan(self.ratio_peak_03),
                    self._replace_nan(self.ratio_peak_12),
                    self._replace_nan(self.ratio_peak_13),
                    self._replace_nan(self.ratio_peak_23),
                    self._replace_nan(self.percentile_10),
                    self._replace_nan(self.percentile_20),
                    self._replace_nan(self.percentile_30),
                    self._replace_nan(self.percentile_40),
                    self._replace_nan(self.percentile_50),
                    self._replace_nan(self.percentile_60),
                    self._replace_nan(self.percentile_70),
                    self._replace_nan(self.percentile_80),
                    self._replace_nan(self.percentile_90),)
        else:
            return self.chaos, self.image_corr, self.pattern_match, self.image_corr_01, self.image_corr_02, self.image_corr_03, self.image_corr_12, self.image_corr_13, self.image_corr_23, self.snr, self.nnz_percent, self.peak_int_diff_0, self.peak_int_diff_1, self.peak_int_diff_2, self.peak_int_diff_3, self.quart_1, self.quart_2, self.quart_3, self.ratio_peak_01, self.ratio_peak_02, self.ratio_peak_0, self.ratio_peak_12, self.ratio_peak_13, self.ratio_peak_23, self.percentile_10, self.percentile_20, self.percentile_30, self.percentile_40, self.percentile_50, self.percentile_60, self.percentile_70, self.percentile_80, self.percentile_90



def get_compute_img_metrics(sample_area_mask, empty_matrix, img_gen_conf):
    """ Returns a function for computing isotope image metrics

    Args
    ------------
    sample_area_mask: ndarray[bool]
        mask for separating sampled pixels (True) from non-sampled (False)
    empty_matrix : ndarray
        empty matrix of the same shape as isotope images
    img_gen_conf : dict
        isotope_generation section of the dataset config
    Returns
    ------------
    : function
        function that returns tuples of metrics for every list of isotope images
    """
    def compute(iso_images_sparse, sf_ints):
        diff = len(sf_ints) - len(iso_images_sparse)
        iso_imgs = [empty_matrix if img is None else img.toarray()
                    for img in iso_images_sparse + [None] * diff]
        iso_imgs_flat = [img.flat[:][sample_area_mask] for img in iso_imgs]

        if img_gen_conf['do_preprocessing']:
            for img in iso_imgs_flat:
                smoothing.hot_spot_removal(img)

        measures = ImgMeasures(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0)
        if len(iso_imgs) > 0:
            measures.pattern_match = isotope_pattern_match(iso_imgs_flat, sf_ints)
            measures.image_corr = isotope_image_correlation(iso_imgs_flat, weights=sf_ints[1:])
            moc = measure_of_chaos(iso_imgs[0], img_gen_conf['nlevels'])
            measures.chaos = 0 if np.isclose(moc, 1.0) else moc
            measures.image_corr_01, measures.image_corr_02, measures.image_corr_03, measures.image_corr_12, \
            measures.image_corr_13, measures.image_corr_23 = isotope_image_correlation_sd(iso_imgs_flat)
            measures.snr = snr_img(iso_imgs[0])
            measures.nnz_percent = percent_nnz(iso_imgs[0])
            measures.peak_int_diff_0, measures.peak_int_diff_1, measures.peak_int_diff_2, measures.peak_int_diff_3 = spectra_int_diff(iso_imgs_flat, sf_ints)
            measures.quart_1, measures.quart_2, measures.quart_3 = quartile_pxl(iso_imgs[0])
            measures.ratio_peak_01, measures.ratio_peak_02, measures.ratio_peak_03, measures.ratio_peak_12, \
            measures.ratio_peak_13, measures.ratio_peak_23 = ratio_peaks(iso_imgs_flat)
            measures.percentile_10, measures.percentile_20, measures.percentile_30, measures.percentile_40, \
            measures.percentile_50, measures.percentile_60, measures.percentile_70, measures.percentile_80, \
            measures.percentile_90 = decile_pxl(iso_imgs[0])

        return measures.to_tuple()

    return compute


def _calculate_msm(sf_metrics_df):
    return sf_metrics_df.chaos * sf_metrics_df.spatial * sf_metrics_df.spectral


def sf_image_metrics(sf_images, sc, formulas, ds, ds_config):
    """ Compute isotope image metrics for each formula

    Args
    ------------
    sc : pyspark.SparkContext
    ds_config : dict
        dataset configuration
    ds : engine.dataset.Dataset
    formulas : engine.formulas.Formulas
    sf_images : pyspark.rdd.RDD
        RDD of (formula, list[images]) pairs
    Returns
    ------------
    : pandas.DataFrame
    """
    nrows, ncols = ds.get_dims()
    empty_matrix = np.zeros((nrows, ncols))
    compute_metrics = get_compute_img_metrics(ds.get_sample_area_mask(), empty_matrix, ds_config['image_generation'])
    sf_add_ints_map_brcast = sc.broadcast(formulas.get_sf_peak_ints())
    # sf_peak_ints_brcast = sc.broadcast(formulas.get_sf_peak_ints())
    colnames = ['sf_id', 'adduct', 'chaos', 'spatial', 'spectral', 'image_corr_01', 'image_corr_02', 'image_corr_03',
                'image_corr_12', 'image_corr_13', 'image_corr_23', 'snr', 'nnz_percent', 'peak_int_diff_0',
                'peak_int_diff_0', 'peak_int_diff_1', 'peak_int_diff_2', 'quart_1', 'quart_2',
                'quart_3', 'ratio_peak_01', 'ratio_peak_02', 'ratio_peak_03', 'ratio_peak_12', 'ratio_peak_13',
                'ratio_peak_23', 'percentile_10', 'percentile_20', 'percentile_30', 'percentile_40', 'percentile_50',
                'percentile_60', 'percentile_70', 'percentile_80', 'percentile_90']

    sf_metrics = (sf_images
                  .map(lambda ((sf, adduct), imgs):
                      (sf, adduct) + compute_metrics(imgs, sf_add_ints_map_brcast.value[(sf, adduct)]))
                  ).collect()
    sf_metrics_df = (pd.DataFrame(sf_metrics, columns=colnames).set_index(['sf_id', 'adduct']))
    sf_metrics_df['msm'] = _calculate_msm(sf_metrics_df)
    return sf_metrics_df


def sf_image_metrics_est_fdr(sf_metrics_df, formulas, fdr):
    sf_msm_df = formulas.get_sf_adduct_sorted_df()
    sf_msm_df = sf_msm_df.join(sf_metrics_df.msm).fillna(0)

    sf_adduct_fdr = fdr.estimate_fdr(sf_msm_df)

    colnames = ['chaos', 'spatial', 'spectral', 'image_corr_01', 'image_corr_02', 'image_corr_03', 'image_corr_12',
                'image_corr_13', 'image_corr_23', 'snr', 'nnz_percent', 'peak_int_diff_0', 'peak_int_diff_1',
                'peak_int_diff_2', 'peak_int_diff_3', 'quart_1', 'quart_2', 'quart_3',
                'ratio_peak_01', 'ratio_peak_02', 'ratio_peak_03', 'ratio_peak_12', 'ratio_peak_13', 'ratio_peak_23',
                'percentile_10', 'percentile_20', 'percentile_30', 'percentile_40', 'percentile_50', 'percentile_60',
                'percentile_70', 'percentile_80', 'percentile_90', 'msm', 'fdr']
    # return sf_metrics_df.join(sf_adduct_fdr, how='inner')[colnames]
    df = sf_metrics_df.join(sf_adduct_fdr, how='outer')#[colnames]
    df = df.fillna(value=-0)
    return df

# def filter_sf_metrics(sf_metrics_df):
#     return sf_metrics_df[(sf_metrics_df.chaos > 0) | (sf_metrics_df.spatial > 0) | (sf_metrics_df.spectral > 0)]
#     # return sf_metrics_df[sf_metrics_df.msm > 0]
#
#
# def filter_sf_images(sf_images, sf_metrics_df):
#     return sf_images.filter(lambda (sf_i, _): sf_i in sf_metrics_df.index)