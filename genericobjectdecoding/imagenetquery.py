"""download tars from image-net.org, access images from said tars."""
import os
import socket
import urllib
import tarfile
import skimage
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("INFO")

IMAGENETURL = "http://www.image-net.org"
IMAGEREQ = "/download/synset?wnid={wnid}&username={username}&accesskey={accesskey}&release=latest&src=stanford"
MAXATTEMPT = 5


def imagefromtar(imname, tarpath, readflag="r", **kwarg):
    with tarfile.open(tarpath, readflag) as archive:
        with archive.extractfile(imname) as imhand:
            im = skimage.io.imread(imhand, **kwarg)
    return im


def downloadtar(wnid, outtar, user=None, accesskey=None):
    request = IMAGENETURL + IMAGEREQ.format(
        wnid=wnid, username=user, accesskey=accesskey
    )
    LOGGER.debug(f"downloading {outtar}")
    # the image-net server is not very good
    for n in range(MAXATTEMPT):
        try:
            urllib.request.urlretrieve(request, filename=outtar)
            LOGGER.debug(f"successful download after {n} attempts")
            return
        except (TimeoutError, urllib.error.URLError):
            LOGGER.warning("server timed out.")
    raise TimeoutError(f"failed to retried {request} in {MAXATTEMPT} attempts")
    return


def query(imagename, outdir, user=None, accesskey=None, **kwarg):
    wnid = imagename.split("_")[0]
    outtar = os.path.join(outdir, wnid + ".tar")
    if not os.path.exists(outtar):
        LOGGER.info(f"{outtar} does not exist, downloading...")
        downloadtar(wnid=wnid, outtar=outtar, user=user, accesskey=accesskey)
    return imagefromtar(imname=imagename, tarpath=outtar, **kwarg)
