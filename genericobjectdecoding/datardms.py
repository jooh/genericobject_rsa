import os
import numpy as np
import pandas as pd

DIRPATTERN = os.path.join("{indir}", "sub-{sub}", "pilab", "sess_{sess}")
FILEPATTERN = "sub-{sub}_mask_{roi}.csv"


def load(**kwarg):
    loaddir = DIRPATTERN.format(**kwarg)
    loadfile = FILEPATTERN.format(**kwarg)
    loadpath = os.path.join(loaddir, loadfile)
    assert os.path.exists(loadpath), f"could not find {loadfile} in {loaddir}"
    df = pd.read_csv(loadpath, sep=",", header=0, index_col=0)
    assert np.all(df.columns == df.index), "input is not valid RDM"
    assert np.allclose(df.values, df.values.T), "input is not symmetrical"
    return df
