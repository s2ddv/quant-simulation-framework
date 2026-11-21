"""Offline daily integration, analytic sampling and bundle integrity."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from forecasting.daily import run_daily, read_daily
from simulation.report import historical_snapshot, SimulationConfig


def snapshot(constant=False):
    returns = np.zeros(80) if constant else np.random.default_rng(17).normal(0,.02,80)
    prices = pd.Series(100*np.exp(np.r_[0,np.cumsum(returns)]),
                       index=pd.bdate_range('2024-01-01', periods=81))
    return historical_snapshot(prices, ticker='SYNTHETIC', period='fixture',
                               retrieved_at='2024-06-01T00:00:00+00:00', source='synthetic')


def test_daily_math_and_sampling():
    config = SimulationConfig(horizon=1/252,steps=1,n_sims=200_000,seed=21)
    report, paths = run_daily(snapshot(), 'historical',window=22,config=config)
    v = report['forecast']['daily_variance']
    s0 = report['last_price']
    assert paths.shape == (2,config.n_sims)
    assert np.all(paths[0] == s0)
    assert report['parameters']['sigma'] == pytest.approx(np.sqrt(252*v))
    assert report['parameters']['mu'] == pytest.approx(126*v)
    mean = s0*np.exp(v/2)
    variance = s0**2*np.exp(v)*np.expm1(v)
    # Six sampling SE for mean; variance/quantile tolerance is generous at N=200k.
    assert abs(paths[-1].mean()-mean) < 6*np.sqrt(variance/config.n_sims)
    assert paths[-1].var(ddof=1) == pytest.approx(variance,rel=.03)
    for p in (.05,.5,.95):
        assert np.quantile(paths[-1],p) == pytest.approx(s0*np.exp(np.sqrt(v)*ndtri(p)),rel=.002)
    losses = s0-paths[-1]
    var = np.quantile(losses,.95)
    assert report['results']['risk']['var'] == pytest.approx(var)
    assert report['results']['risk']['cvar'] == pytest.approx(losses[losses>=var].mean())
    assert var == pytest.approx(report['results']['analytical_var'], rel=.02)
    _, again = run_daily(snapshot(),'historical',window=22,config=config)
    assert np.array_equal(paths,again)


@pytest.mark.parametrize('model', ['historical','ewma','ml'])
def test_daily_zero(model):
    report, paths = run_daily(snapshot(True),model)
    assert np.all(paths == 100)
    assert report['results']['risk']['var'] == 0
    assert report['results']['risk']['cvar'] == 0


@pytest.mark.parametrize('horizon,steps', [(1,1),(1/252,2)])
def test_reject_multisession(horizon,steps):
    with pytest.raises(ValueError,match='requires'):
        run_daily(snapshot(),'ewma',config=SimulationConfig(horizon=horizon,steps=steps))


def test_replay_and_tampering(tmp_path):
    report, paths = run_daily(snapshot(),'ewma',factor=.9)
    path = tmp_path/'saved.json'
    path.write_text(json.dumps(report))
    replay, replay_paths = read_daily(path)
    assert replay == report
    assert np.array_equal(paths,replay_paths)
    for target in ('results','historical_data','forecast'):
        altered = copy.deepcopy(report)
        if target == 'results':
            altered[target]['risk']['var'] += 1
        elif target == 'historical_data':
            altered[target]['observations'][-1]['close'] += 1
        else:
            altered[target]['daily_variance'] += 1
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            read_daily(path)


def test_cli_bundle_offline_and_no_overwrite(tmp_path,monkeypatch):
    import main
    def forbidden(*a,**kw):
        raise AssertionError('provider must not be used')
    monkeypatch.setattr(main,'download_prices',forbidden)
    monkeypatch.setattr(main,'download_company',forbidden)
    source = tmp_path/'snapshot.json'
    source.write_text(json.dumps(snapshot()))
    destination = tmp_path/'daily.json'
    args = ['--next-session','--snapshot',str(source),'--model','ml','--no-plot','--report',str(destination)]
    main.main(args)
    bundle = destination.with_suffix('')
    assert {p.name for p in bundle.iterdir()} == {'report.json','paths.png','convergence.png','final-distribution.png'}
    with pytest.raises(FileExistsError):
        main.main(args)
    main.main(['--next-session','--replay',str(bundle/'report.json'),'--no-plot',
               '--report',str(tmp_path/'replay.json')])


def test_publication_failure_leaves_no_bundle(tmp_path,monkeypatch):
    import main
    report, paths = run_daily(snapshot(),'historical')
    def fail(*a,**kw):
        raise RuntimeError('plot failure')
    monkeypatch.setattr(main,'plot_paths',fail)
    with pytest.raises(RuntimeError,match='plot failure'):
        main.publish_run(report,paths,SimulationConfig(**report['simulation']),tmp_path/'failed.json',True)
    assert list(tmp_path.iterdir()) == []
