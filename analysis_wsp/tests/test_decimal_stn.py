"""Scientific/regression checks; no official experiment is executed by tests."""
from pathlib import Path
import math

import pytest

from analysis_wsp.layout import ROOT
from analysis_wsp.ibs_decimal_pipeline.common import read_csv
from analysis_wsp.ibs_decimal_pipeline.model import centi, load_model, evaluate, signature_allocations, validate_trajectory
from analysis_wsp.ibs_decimal_pipeline.networks import network, ordered


def test_exact_decimal_and_precision_rejection():
    assert centi('123.13') == 12313
    assert centi('312.60') == 31260
    assert centi('0.01') == 1
    assert centi('123') == 12300
    for value in ('12.345','-0.01','NaN','Infinity'):
        with pytest.raises(ValueError):
            centi(value)


def test_horizon_boundary_and_resource_validation():
    # Strict objective convention: arrival exactly at H does not burn before H.
    model = {'n':3,'horizon':200,'delay':100,'ignition':0,'releases':{50:1},
             'adjacency':[[(1,100)],[(2,100)],[]]}
    assert evaluate(model,[])[0] == 2
    assert evaluate(model,[(1,50)])[0] == 2
    for allocation in ([(1,49)],[(1,50),(2,50)],[(1,50),(1,50)],[(8,50)]):
        with pytest.raises(ValueError):evaluate(model,allocation)
    model['releases']={150:1}
    with pytest.raises(ValueError,match='after fire arrival'):
        evaluate(model,[(1,150)])


def test_real_archived_small_example_and_truncated_signature_rejected(tmp_path):
    name='ga_small_05e4983100f5eeb1'
    path=ROOT/'data/instances/harris'/f'{name}.json'
    original='25@1323-44@944-65@881-86@628-107@565-128@502-149@375-170@123-171@312-172@438'
    corrected='25@1323.15-44@944.20-65@881.04-86@628.40-107@565.24-128@502.08-149@375.76-170@123.13-171@312.60-172@438.92'
    assert evaluate(load_model(path,True),signature_allocations(original,True))[0]==391
    assert evaluate(load_model(path),signature_allocations(corrected))[0]==390
    with pytest.raises(ValueError,match='invalid release time'):
        evaluate(load_model(path),signature_allocations(original))
    csv=tmp_path/'trajectory.csv'
    csv.write_text(f'Run,Fitness,Solution\n1,390,{corrected}\n')
    assert validate_trajectory(csv,load_model(path),1)[0]['best_objective']==390
    csv.write_text(f'Run,Fitness,Solution\n1,391,{corrected}\n')
    with pytest.raises(ValueError,match='disagrees'):
        validate_trajectory(csv,load_model(path),1)


def test_network_transitions_do_not_cross_seeds():
    row=lambda seed,fit,sig:dict(Run=str(seed),Fitness=str(fit),Solution=sig)
    runs=[(1,[row(1,3,'a'),row(1,2,'b'),row(1,2,'b')]),
          (2,[row(2,4,'c'),row(2,2,'b')])]
    metrics,graph=network('test',runs,'RAW')
    assert {(e['source'],e['target']) for e in graph['edges']}=={('a','b'),('c','b')}
    assert metrics['transition_count']==2
    assert metrics['best_strength_uncapped']==1
    assert metrics['endpoint_best_fraction']==1
    assert metrics['mean_pairwise_jaccard']==pytest.approx(1/3)
    rows=[row(1,5,'a'),row(1,7,'c'),row(1,5,'b'),row(1,4,'d')]
    assert [r['Solution'] for r in ordered(rows)]==['a','b','d']
