from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pygad
import pytest
from instancespace.data.options import SiftedOptions
from instancespace.stages.sifted import SiftedStage
from analysis_wsp.matilda74 import sifted_correction as fix


def test_negative_worst_error_is_maximized(monkeypatch):
    state = SimpleNamespace(
        selfx=np.arange(24).reshape(12, 2), selfy=np.zeros((12, 2)),
        selfy_bin=np.zeros((12, 2)), clust=np.ones((2, 1), dtype=bool),
        selffeat_labels=np.array(["a", "b"]), cv_partition=None,
    )
    monkeypatch.setattr(fix.PilotStage, "pilot", lambda *a: SimpleNamespace(z=np.zeros((12, 2))))
    fitness=[]
    for losses in ([.1,.2], [.1,.4]):
        responses=iter(losses)
        monkeypatch.setattr(fix,"cross_val_score",lambda *a,**kw: np.full(5,-next(responses)))
        fitness.append(fix.cost_fcn(state,np.array([0]),0))
    assert fitness==[-.2,-.4]
    ga=pygad.GA(num_generations=1,num_parents_mating=1,initial_population=np.array([[0],[1]]),
                fitness_func=lambda ga,x,i:0,mutation_type=None,crossover_type=None,suppress_warnings=True)
    assert ga.best_solution(pop_fitness=np.array(fitness))[2]==0


def test_correlation_keeps_actual_significant_absolute_correlations():
    rho=np.array([[.01],[-.6],[.9],[.02]])
    p=np.full_like(rho,.001)
    state=SimpleNamespace(
        x=np.arange(80).reshape(20,4),y=np.zeros((20,1)),
        feat_labels=np.array(["f0","f1","f2","f3"]),
        opts=SimpleNamespace(rho=.5),PVAL_THRESHOLD=.05,
        _compute_correlation=lambda x,y:(rho.copy(),p.copy()),
    )
    old=SiftedStage.select_features_by_performance(state)[3]
    corrected=fix.select_features_by_performance(state)
    assert old.tolist()==[2,3]  # wrong low-correlation column; misses negative correlate
    assert corrected[3].tolist()==[1,2]
    np.testing.assert_array_equal(corrected[0],state.x[:,[1,2]])
    np.testing.assert_array_equal(corrected[1],rho)
    p[1]=.2
    assert fix.select_features_by_performance(state)[3].tolist()==[2]


def test_filtered_candidate_and_final_columns_identical(monkeypatch):
    captured=[]
    class FakeGA:
        def __init__(self,**kwargs):
            self.generations_completed=1
            self.best_solutions_fitness=[-.2,-.2]
        def run(self):
            captured.append((self.selfx.copy(),np.asarray(self.selffeat_labels).copy()))
        def best_solution(self):
            return np.array([0,1]),-.2,0
    monkeypatch.setattr(fix.pygad,"GA",FakeGA)
    monkeypatch.setattr(fix,"cost_fcn",lambda *a:-.2)
    state=SimpleNamespace(
        x=np.arange(72).reshape(12,6),y=np.zeros((12,1)),y_bin=np.zeros((12,1)),
        feat_labels=np.array(["f0","f1","f2","f3","f4","f5"]),
        opts=replace(SiftedOptions.default(),k=2),
        parallel_options=SimpleNamespace(flag=False,n_cores=1),
    )
    filtered=np.array([1,4,5]); x_aux=state.x[:,filtered]
    clust=np.array([[1,0],[0,1],[0,1]],dtype=bool)
    # Run original mapping with a fake optimizer: it supplies all six columns.
    SiftedStage._find_best_combination(state,x_aux,clust,filtered,np.random.default_rng(0))
    assert captured[0][0].shape[1]==6
    selected,indices=fix.find_best_combination(state,x_aux,clust,filtered,np.random.default_rng(0))
    np.testing.assert_array_equal(captured[1][0],x_aux)
    assert captured[1][1].tolist()==["f1","f4","f5"]
    assert indices.tolist()==[1,5]
    np.testing.assert_array_equal(selected,state.x[:,[1,5]])


def test_misaligned_cost_input_fails():
    state=SimpleNamespace(selfx=np.zeros((10,6)),clust=np.ones((3,2),dtype=bool))
    with pytest.raises(ValueError,match="misaligned"):
        fix.cost_fcn(state,np.array([0,0]),0)


def test_context_restores_upstream_and_records_identity(tmp_path):
    original=SiftedStage.cost_fcn
    path=tmp_path/"trace.json"
    with fix.corrected_sifted(path):
        assert SiftedStage.cost_fcn is fix.cost_fcn
    assert SiftedStage.cost_fcn is original
    record=json.loads(path.read_text())
    assert record["status"]=="completed"
    assert record["implementation"]["revision"]==fix.REVISION
