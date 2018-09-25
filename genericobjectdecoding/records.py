import os
import argparse
import tensorflow as tf
import logging
import numpy as np
from genericobjectdecoding import util
from genericobjectdecoding import imagenetquery

PXSIZE = 500
FINALSIZE = 550
DEGSIZE = 12.
PX2DEG = DEGSIZE / PXSIZE
DEG2PX = PXSIZE / DEGSIZE
PARSER = argparse.ArgumentParser(
    "Download and pre-process imagenet exemplars, save as TFRecords."
)
PARSER.add_argument(
    "--csvpath",
    required=True,
    help="path to csv file (e.g., something like [your root]/GenericObjectDecoding/data/imageURL_test.csv )",
)
PARSER.add_argument("--directory", default=os.getcwd(), help="output directory")
PARSER.add_argument("--user", help="imagenet username")
PARSER.add_argument("--accesskey", help="imagenet accesskey")
PARSER.add_argument(
    "-ll",
    "--loglevel",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default="INFO",
    help="logging level",
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(PARSER.get_default("loglevel"))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def prepareimage(name, directory, user, accesskey):
    """prepare images similarly to how they were presented in the scanner"""
    # load (probably from URL)
    im = imagenetquery.query(
        imagename=name.decode("utf-8"),
        outdir=directory.decode("utf-8"),
        user=user.decode("utf-8"),
        accesskey=accesskey.decode("utf-8"),
        as_grey=True,
    )
    im = util.squarecrop(im)
    # in principle we could do this with tf.image, but
    # there are known bugs in TF's resize at the moment -
    # https://github.com/tensorflow/tensorflow/issues/6720
    im = util.resize(im, [PXSIZE, PXSIZE])
    finalim = np.ones([FINALSIZE, FINALSIZE], dtype="float32") * .5
    finalim[util.indexcenter(finalim, PXSIZE)] = im
    return np.reshape(finalim, [FINALSIZE, FINALSIZE, 1])


def makeexample(name, im):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={"image": _bytes_feature(im), "imagename": _bytes_feature(name)}
        )
    )
    return example.SerializeToString()


def parsecsv(row):
    return tf.decode_csv(row, [[tf.constant("")]] * 3, field_delim=",")[-1]


def decodeexample(serialised):
    """decode a single serialised example. Used primarily to map from
    TFRecordDataset to a Dataset with loaded and prepped images """
    features = tf.parse_single_example(
        serialised,
        features={
            "image": tf.FixedLenFeature([], tf.string),
            "imagename": tf.FixedLenFeature([], tf.string),
        },
    )
    image = tf.image.decode_png(features["image"], channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # center on mid-gray
    image -= .5
    return image, features["imagename"]


def loaddata(tfrecordpath):
    """return a tensor dataset with decoded, prepared images and
    labels."""
    data = tf.data.TFRecordDataset(tfrecordpath)
    return data.map(decodeexample)


def encodedata(csvpath, directory=None, user=None, accesskey=None):
    namebase = os.path.splitext(os.path.split(csvpath)[1])[0]
    tfrecordpath = os.path.join(directory, namebase + ".tfrecords")
    assert not os.path.exists(tfrecordpath), f"{tfrecordpath} already exists"
    LOGGER.info(f"initialised record file at {tfrecordpath}")
    tfwriter = tf.python_io.TFRecordWriter(tfrecordpath)
    # graph
    filedata = tf.data.TextLineDataset(csvpath)
    iterator = filedata.make_one_shot_iterator()
    imagename = parsecsv(iterator.get_next())
    image = tf.py_func(
        prepareimage,
        [imagename, directory, user, accesskey],
        tf.float32,
        stateful=False,
        name="prepareimage",
    )
    imagebytes = tf.image.encode_png(tf.image.convert_image_dtype(image, tf.uint16))
    # /graph
    niter = 0
    nfail = 0
    sess = tf.Session()
    while True:
        try:
            resdict = sess.run(
                {"image": image, "name": imagename, "imagebytes": imagebytes}
            )
            if np.any(np.isnan(resdict["image"])):
                LOGGER.warning(f'failed to locate {resdict["name"]}')
                nfail += 1
                continue
            sstring = makeexample(resdict["name"], resdict["imagebytes"])
            tfwriter.write(sstring)
            LOGGER.info(
                "saved {name} successfully as record {niter}".format(
                    name=resdict["name"], niter=niter
                )
            )
            niter += 1
        except tf.errors.OutOfRangeError:
            LOGGER.info("finished iterating")
            break
    LOGGER.info(f"saved {niter} images successfully")
    LOGGER.info(f"{nfail} images failed to convert")
    sess.close()
    tfwriter.close()
    return tfrecordpath


if __name__ == "__main__":
    inarg = PARSER.parse_args()
    LOGGER.setLevel(inarg.loglevel)
    imagenetquery.LOGGER.setLevel(inarg.loglevel)
    tfrecordpath = encodedata(
        csvpath=inarg.csvpath,
        directory=inarg.directory,
        user=inarg.user,
        accesskey=inarg.accesskey,
    )
    LOGGER.info(f"saved records to {tfrecordpath}")
