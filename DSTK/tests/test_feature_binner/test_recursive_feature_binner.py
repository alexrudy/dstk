import sklearn.datasets as ds
import numpy as np
from DSTK.FeatureBinning import decision_tree_binner as tfb
import re

cancer_ds = ds.load_breast_cancer()
data = cancer_ds['data']
target = cancer_ds['target']


def test_recursion():
    binner = tfb.DecisionTreeBinner('test', max_leaf_nodes=4)
    binner.fit(data[:, 0], target)

    np.testing.assert_allclose(binner.splits, [13.094999313354492, 15.045000076293945, 16.924999237060547, np.PINF, np.NaN])
    np.testing.assert_allclose(binner.values, [[0.04905660377358491, 0.9509433962264151],
                                            [0.2878787878787879, 0.7121212121212122],
                                            [0.8148148148148148, 0.18518518518518517],
                                            [0.9915254237288136, 0.00847457627118644],
                                            [0.37258347978910367, 0.62741652021089633]])


def test_recursion_with_mdlp():
    binner = tfb.DecisionTreeBinner('test', mdlp=True)
    binner.fit(data[:, 0], target)

    np.testing.assert_allclose(binner.splits, [13.094999313354492, 15.045000076293945, 17.880001068115234, np.PINF, np.NaN])
    np.testing.assert_allclose(binner.values, [[0.04905660377358491, 0.9509433962264151],
                                            [0.2878787878787879, 0.7121212121212122],
                                            [0.8533333333333334, 0.14666666666666667],
                                            [1.0, 0.0],
                                            [0.37258347978910367, 0.62741652021089633]])

assert_pat = \
r"""<= 13.094999[\d]+: \[ 0.049[\d]+  0.950[\d]+\]
<= 15.045000[\d]+: \[ 0.2878[\d]+  0.7121[\d]+\]
<= 17.880001[\d]+: \[ 0.8533[\d]+  0.1466[\d]+\]
<= inf: \[ 1.[\d]*  0.[\d]*\]
NaN: \[ 0.3725[\d]+  0.6274[\d]+\]"""


def test_str_repr_with_mdlp():

    binner = tfb.DecisionTreeBinner('test', mdlp=True)
    binner.fit(data[:, 0], target)

    assert re.match(assert_pat, str(binner))


def test_fit():
    feats = [0, 1, 2, np.nan, 5, np.nan]
    labels = [0, 1, 0, 1, 1, 0]

    binner = tfb.DecisionTreeBinner('test', max_leaf_nodes=3)
    binner.fit(feats, labels)

    np.testing.assert_equal(binner.splits, [0.5, 1.5, np.PINF, np.NaN])
    np.testing.assert_equal(binner.values, [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5]])


def test_transform():
    feats = [0, 1, 2, np.nan, 5, np.nan]
    labels = [0, 1, 0, 1, 1, 0]

    binner = tfb.DecisionTreeBinner('test', max_leaf_nodes=3)

    binner.fit(feats, labels)
    np.testing.assert_equal(binner.transform(feats, class_index=1), [0.0, 1.0, 0.5, 0.5, 0.5, 0.5])


def test_fit_transform():
    feats = [0, 1, 2, np.nan, 5, np.nan]
    labels = [0, 1, 0, 1, 1, 0]

    binner = tfb.DecisionTreeBinner('test', max_leaf_nodes=3)
    trans = binner.fit_transform(feats, labels, class_index=1)
    np.testing.assert_equal(trans, [0.0, 1.0, 0.5, 0.5, 0.5, 0.5])
