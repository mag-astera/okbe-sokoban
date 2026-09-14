"""Range predicates must preserve exact boundaries and product dynamics."""
import copy
import itertools
import numpy as np
import pytest
from test_dbn_composer import setup
import dbn_composer as dbn


def test_nested_ranges_features_and_complement():
    dims = {'x': 12, 'y': 8, 'z': 2, 'action': 6}
    feature = {'band': {'var':'x', 'gt':5, 'lt':9}}
    expr = {'and':[{'feature':'band'}, {'or':[{'var':'y','gte':6}, {'not':{'var':'z','in':[0]}}]}]}
    pred = dbn.gate(expr, dims, feature)
    for x,y,z in itertools.product(range(12),range(8),range(2)):
        assert pred.eval(dict(x=x,y=y,z=z),0) == (5<x<9 and (y>=6 or z!=0))
    assert pred.vars() == {'x','y','z'}


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), '5', None])
def test_invalid_comparison_bounds(value):
    with pytest.raises(ValueError, match='finite numbers'):
        dbn.gate({'var':'x','gt':value}, {'x':10}, {})


def test_full_product_ranges_match_membership_with_identity():
    spaces, spec = setup()
    spec['features']['trigger'] = {'and':[{'var':'X','gt':0,'lte':1}, {'var':'action','gte':0}]}
    spec['x_identity_when'] = {'or':[{'var':'A','lt':1}, {'var':'B','gte':1}]}
    reference = copy.deepcopy(spec)
    reference['features']['trigger'] = {'and':[{'var':'X','in':[1]}, {'var':'action','in':list(range(spaces['X'].na))}]}
    reference['x_identity_when'] = {'or':[{'var':'A','in':[0]}, {'var':'B','in':[1]}]}
    actual, expected = {}, {}
    dbn.evaluate(spaces,spec,build=True,retain=actual)
    dbn.evaluate(spaces,reference,build=True,retain=expected)
    for a,b in zip(actual['csr'],expected['csr']):
        np.testing.assert_allclose(a.toarray(),b.toarray(),rtol=0,atol=0)
