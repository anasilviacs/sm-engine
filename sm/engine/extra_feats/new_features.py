import numpy as np
from scipy import stats
# from pyIMS.imutils import nan_to_zero

# TODO add more features - histogram kde, connected elements

def isotope_image_correlation_sd(images_as_2darray):
    """
    Sven's function; correlation between pairs of iso images: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

    :param images_as_2darray: m by n matrix with m images of n pixels each, each one corresponding to one iso peak
    :return pearson correlation between pairs of images. 10 if one of the images is not present (nan not supported when
    writing the json to the DB)
    :rtype list with 6 elements (float)
    """

    while len(images_as_2darray) < 4:
        images_as_2darray.append([0])
    buf = []
    for i in range(4):
        for j in range(i + 1, 4):
            if len(images_as_2darray[i]) == 1:
                buf.append(0)
                continue
            if len(images_as_2darray[j]) == 1:
                buf.append(0)
                continue
            cor = stats.pearsonr(images_as_2darray[i], images_as_2darray[j])[0]
            if np.isnan(cor):
                cor = 0
            buf.append(cor)
    return buf


def snr_img(image):
    """
    Signal-to-Noise Ratio for one image

    :param image: first isotopic image as 2d array
    :return signal to noise ration for the given image
    :rtype float
    """
    snr = float(stats.signaltonoise(image, axis=None))
    return snr


def percent_zero(image):
    """
    Percentage of zero values in one image

    :param image: first isotopic image as 2d array
    :return number of non-zero pixels over the total number of pixels
    :rtype float
    """

    percent = 1.0 - (float(np.count_nonzero(image)) / float(image.shape[0] * image.shape[1]))
    return percent


def spectra_int_diff(images_flat, theor_iso_intensities):
    """
    This function calculates the difference in intensity between the first 5 theoretical and measured peaks.

    :param images_flat: 2d array (or sequence of 1d arrays) of pixel intensities with shape (d1, d2) where d1 is the
        number of images and d2 is the number of pixels per image, i.e. images_flat[i] is the i-th flattened image
    :param theor_iso_intensities: 1d array (or sequence) of theoretical isotope intensities with shape d1, i.e.
        theor_iso_intensities[i] is the theoretical isotope intensity corresponding to the i-th image
    :return difference between the measured and theoretic peaks' intensity
    :rtype list with 5 elements (float)
    """

    image_ints = []
    # not_null = images_flat > 0
    for ii, _ in enumerate(theor_iso_intensities):
        images_flat[ii][np.isnan(images_flat[ii])] = 0
        image_ints.append(np.sum(images_flat[ii]))
        # image_ints.append(np.sum(images_flat[ii][not_null[ii]]))

    image_ints /= np.linalg.norm(image_ints)
    theor_iso_intensities /= np.linalg.norm(theor_iso_intensities)
    int_diff = image_ints - theor_iso_intensities

    while len(int_diff) < 4:
        int_diff = np.append(int_diff, 2.01) # in the case where one of the peaks is missing the difference is maximum

    return int_diff[:4]


def quartile_pxl(image):
    """
    computes the normalized intensities that split the image into quartiles (in terms of number of pixels)

    :param image: one image as a 2d array
    :return intensities (after normalization) that split the number of pixels into quartiles
    :rtype 1 by 3 array of floats
    """
    image /= np.linalg.norm(image)
    quart = stats.mstats.mquantiles(image)

    return quart


def decile_pxl(image):
    """
    computes the normalized intensities that split the image into deciles (in terms of number of pixels)

    :param image: one image as a 2d array
    :return intensities (after normalization) that split the number of pixels into deciles
    :rtype 1 by 9 array of floats
    """
    image /= np.linalg.norm(image)
    decile = stats.mstats.mquantiles(image, prob=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])

    return decile


def ratio_peaks(images_flat):
    """
    This function calculates the ratio in intensity between the measured isotopic peaks.

    :param images_flat: 2d array (or sequence of 1d arrays) of pixel intensities with shape (d1, d2) where d1 is the
        number of images and d2 is the number of pixels per image, images_flat[i] is the i-th flattened image
    :return: difference between the measured and theoretic peaks' intensity
    :rtype: list with 6 elements (float)
    """

    image_ints = []
    for ii in range(0, len(images_flat)):
        image_ints.append(np.sum(images_flat[ii]))
    while len(image_ints) < 4:
        image_ints.append(0)

    image_ints /= np.linalg.norm(image_ints)

    buf = []
    for i in range(4):
        for j in range(i + 1, 4):
            ratio = image_ints[i] / float(image_ints[j])
            if np.isnan(ratio) or np.isinf(ratio):
                ratio = 0
            buf.append(ratio)

    return buf
