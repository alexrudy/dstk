import sklearn.datasets as ds
from DSTK.FeatureBinning.conditional_inference_binner import ConditionalInferenceBinner
import numpy as np
import pandas as pd
import re

cancer_ds = ds.load_breast_cancer()
cancer_data = cancer_ds['data']
cancer_target = cancer_ds['target']

cancer_df = pd.DataFrame(cancer_data, columns=cancer_ds['feature_names'])


assert_pat = \
r"""<= 11.75: \[ 0.02  0.98\]
<= 13.07999[\d]+: \[ 0.08695[\d]+  0.91304[\d]+\]
<= 15.039999[\d]+: \[ 0.28787[\d]+  0.71212[\d]+\]
<= 16.84000[\d]+: \[ 0.8148[\d]+  0.1851[\d]+\]
<= inf: \[ 0.9915[\d]+  0.0084[\d]+\]
NaN: \[ 0.3725[\d]+  0.6274[\d]+\]"""


def test_recursion():
    col = 'mean radius'
    data = cancer_df[col].values
    cib = ConditionalInferenceBinner('test_dim_{}'.format(col), alpha=0.95)
    cib.fit(data, cancer_target)

    np.testing.assert_allclose(cib.splits, [11.75, 13.079999923706055, 15.039999961853027, 16.84000015258789, np.PINF, np.NaN])
    np.testing.assert_allclose(cib.values,
                            [[0.02, 0.97999999999999998],
                             [0.086956521739130432, 0.91304347826086951],
                             [0.2878787878787879, 0.71212121212121215],
                             [0.81481481481481477, 0.18518518518518517],
                             [0.99152542372881358, 0.0084745762711864406],
                             [0.37258347978910367, 0.62741652021089633]])


def test_string_repr():
    col = 'mean radius'
    data = cancer_df[col].values
    cib = ConditionalInferenceBinner('test_dim_{}'.format(col), alpha=0.95)
    cib.fit(data, cancer_target)

    assert re.match(assert_pat, str(cib))


def test_adding_bin():
    col = 'mean radius'
    data = cancer_df[col].values
    cib = ConditionalInferenceBinner('test_dim_{}'.format(col), alpha=0.95)
    cib.fit(data, cancer_target)

    cib.add_bin(-1.0, [0.1, 0.9])

    np.testing.assert_allclose(cib.splits, [-1.0, 11.75, 13.079999923706055, 15.039999961853027, 16.84000015258789, np.PINF, np.NaN])
    np.testing.assert_allclose(cib.values,
                            [[0.1, 0.9],
                             [0.02, 0.97999999999999998],
                             [0.086956521739130432, 0.91304347826086951],
                             [0.2878787878787879, 0.71212121212121215],
                             [0.81481481481481477, 0.18518518518518517],
                             [0.99152542372881358, 0.0084745762711864406],
                             [0.37258347978910367, 0.62741652021089633]])


def test_adding_bin_with_non_numeric_splits_only():
    cib = ConditionalInferenceBinner('test', alpha=0.05)
    cib.splits = [np.PINF, np.NaN]
    cib.values = [[0.1, 0.9], [0.8, 0.2]]
    cib.is_fit = True

    cib.add_bin(-1.0, [0.3, 0.7])
    np.testing.assert_allclose(cib.splits, [-1.0, np.PINF, np.NaN])
    np.testing.assert_allclose(cib.values, [[0.3, 0.7], [0.1, 0.9], [0.8, 0.2]])


def test_recursion_with_nan():
    col = 'mean area'
    data = cancer_df[col].values
    rand_idx = np.linspace(1, 500, 23).astype(int)
    data[rand_idx] = np.NaN

    cib = ConditionalInferenceBinner('test_dim_{}'.format(col), alpha=0.95)
    cib.fit(data, cancer_target)

    np.testing.assert_allclose(cib.splits, [471.29998779296875, 555.0999755859375, 693.7000122070312, 880.2000122070312, np.PINF, np.NaN])
    np.testing.assert_allclose(cib.values,
                            [[0.030769230769230771, 0.96923076923076923],
                             [0.13414634146341464, 0.86585365853658536],
                             [0.31730769230769229, 0.68269230769230771],
                             [0.83333333333333337, 0.16666666666666666],
                             [0.99145299145299148, 0.0085470085470085479],
                             [0.2608695652173913, 0.73913043478260865]])


def test_recursion_with_nan_and_special_value():
    col = 'mean area'
    data = cancer_df[col].values
    rand_idx = np.linspace(1, 500, 23).astype(int)
    data[rand_idx] = np.NaN

    rand_idx_2 = np.linspace(1, 550, 29).astype(int)
    data[rand_idx_2] = -1.0

    cib = ConditionalInferenceBinner('test_dim_{}'.format(col), alpha=0.95, special_values=[-1.0, np.NaN])
    cib.fit(data, cancer_target)

    np.testing.assert_allclose(cib.splits, [-1.0, 471.29998779296875, 572.2999877929688, 693.7000122070312, 819.7999877929688, np.PINF, np.NaN])
    np.testing.assert_allclose(cib.values,
                            [[0.4827586206896552, 0.5172413793103449],
                             [0.032432432432432434, 0.9675675675675676],
                             [0.14432989690721648, 0.8556701030927835],
                             [0.3132530120481928, 0.6867469879518072],
                             [0.8205128205128205, 0.1794871794871795],
                             [1.0, 0.0],
                             [0.23809523809523808, 0.7619047619047619]])
