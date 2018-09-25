"""utility functions for gabor model - image processing mainly"""

import urllib
import numpy as np
import skimage
import skimage.io
import skimage.transform


def resize(im, newsize):
    return skimage.transform.resize(im, newsize, order=3, mode="edge").astype("float32")


def squarecrop(im):
    return im[indexcenter(im, min(im.shape))]


def indexcenter(x, n):
    return np.ix_(
        *(
            np.arange(np.ceil(-n / 2), np.ceil(n / 2))[None, :]
            + np.floor(np.array(x.shape)[:, None] / 2)
        ).astype(int)
    )


def indexmiddle(x, n, axis=0):
    """return a slice for the central indices along the dimension axis of x."""
    return (n * np.array([-.5, .5]) + (x.shape[axis] / 2.)).astype(int)


def fsize(theta=0., sigma=None, n_stds=2.):
    """expected size of gabor filter. Use to work out what set of parameters
    will produce a filter with at least the required size (then trim to final
    size with gabortrim)"""
    return 1 + int(
        2
        * np.ceil(
            np.max(np.abs(n_stds * sigma * np.array([np.sin(theta), np.cos(theta)])))
        )
    )


def imexpand(im, sz, grayval=.5):
    """expand image. Probably superceded by np.pad, but the call syntax there is
    much more awkward."""
    imdim = np.shape(im)
    if np.ndim(im) != 2:
        raise ValueError("input im must be 2D")
    if np.any(np.array(sz) < imdim):
        raise ValueError("sz must be equal or greater than shape(im)")
    imageexp = np.ones(sz, dtype=im.dtype) * grayval
    rowind = indexmiddle(imageexp, imdim[0], 0)
    colind = indexmiddle(imageexp, imdim[1], 1)
    imageexp[rowind[0] : rowind[-1], colind[0] : colind[-1]] = im
    return imageexp


def imloader(url):
    """load image from url and return numpy array - suitable for use as
    tf.py_func. url is a byte string that can be decoded as UTF-8."""
    # convert from byte string, return at tf precision
    try:
        im = skimage.io.imread(url.decode("UTF-8"))
        im = skimage.color.rgb2gray(im).astype("float32")
        # stop rounding error causing overflow
        im /= im.max()
        assert im.min() >= 0. and im.max() <= 1., "image outside grayscale range"
        assert im.ndim == 2, "grayscale conversion failed"
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        im = None
    return im


def gwpfilters(sigma, nsigma, cyclespersigma=.5, norient=8, nphase=2, mode="pad"):
    orientations = np.linspace(0, np.pi, norient + 1)[:-1]
    phasevalues = np.linspace(0, np.pi / 2., nphase)
    k = [
        gaborweights(
            frequency=cyclespersigma / sigma,
            theta=direction,
            sigma_x=sigma,
            sigma_y=sigma,
            offset=phase,
            n_stds=nsigma,
        )
        for direction in orientations
        for phase in phasevalues
    ]
    # ah, but they won't be the same size...
    if mode == "crop":
        newdim = np.array([thisk.shape for thisk in k]).min()
        newk = []
        for thisk in k:
            ind = indexmiddle(thisk, newdim, axis=0)
            newk.append(thisk[ind[0] : ind[-1], ind[0] : ind[-1]])
            newk[-1] = np.reshape(newk[-1], list(newk[-1].shape) + [1])
        # tf expects the filter tensor to be h * w * inchannel * outchannel
        return np.stack(newk, axis=3)
    if mode == "pad":
        return hardstack(k)
    else:
        raise ValueError("mode must be pad or stack")


def hardstack(k, grayval=0.):
    dim = np.array([thisk.shape for thisk in k])
    # nb ndim is 1-based so this is actually the index for dim end+2
    stackax = k[0].ndim + 1
    newdim = np.max(dim, axis=0)
    for kind, thisk in enumerate(k):
        lpad = np.floor((newdim - np.array(thisk.shape)) / 2).astype(int)
        rpad = np.ceil((newdim - np.array(thisk.shape)) / 2).astype(int)
        finalpad = tuple(zip(lpad, rpad))
        k[kind] = np.pad(thisk, finalpad, "constant", constant_values=grayval)
        k[kind] = np.reshape(k[kind], list(k[kind].shape) + [1])
    return np.stack(k, axis=stackax)


# will be useful later
def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y
