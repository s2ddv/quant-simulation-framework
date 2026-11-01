import json
import numpy as np
import pandas as pd
import pytest
from forecasting.study import summarize, write_new, compute
from forecasting.core import ForecastConfig
from forecasting.report import run_experiment
from simulation.report import historical_snapshot


def test_summary_pairing_and_segments():
    prices = pd.Series(np.exp(np.cumsum(np.random.default_rng(4).normal(0,.01,200))), index=pd.bdate_range('2018-01-01',periods=200))
    snapshot = historical_snapshot(prices,ticker='TEST',period='test',retrieved_at='2026-01-01T00:00:00+00:00')
    result = run_experiment(snapshot,ForecastConfig(train_size=30,validation_size=20))
    rows=result['experiment']['test']['forecasts']
    summary=summarize(rows)
    assert summary['count']==len(rows)
    assert summary['bootstrap']['10']['mean'][0]==pytest.approx(summary['metrics']['ml']['mse']-summary['metrics']['historical']['mse'])
    assert summarize([])['count']==0


def test_write_refuses_overwrite(tmp_path):
    path=tmp_path/'data.json'
    write_new(path,{'a':1})
    with pytest.raises(FileExistsError):
        write_new(path,{'a':2})
    assert json.loads(path.read_text())=={'a':1}


def test_manifest_integrity_before_evaluation(tmp_path):
    write_new(tmp_path/'manifest.json',{'protocol_sha256':'bad'})
    (tmp_path/'protocol.md').write_text('changed')
    with pytest.raises(ValueError,match='Protocol checksum'):
        compute(tmp_path)

@pytest.fixture
def frozen_bundle(tmp_path):
    from hashlib import sha256
    dates = pd.bdate_range('2015-01-01', periods=80)
    for year in (2018, 2019, 2020, 2022, 2024):
        dates = dates.append(pd.bdate_range(f'{year}-01-01', periods=25))
    prices = pd.Series(np.exp(np.cumsum(np.random.default_rng(7).normal(0,.01,len(dates)))), index=dates)
    snapshot = historical_snapshot(prices,ticker='SPY',period='test',retrieved_at='2026-01-01T00:00:00+00:00')
    write_new(tmp_path/'SPY.json', snapshot)
    (tmp_path/'protocol.md').write_bytes(b'fixed protocol\n')
    manifest = {'protocol_sha256':sha256((tmp_path/'protocol.md').read_bytes()).hexdigest(),
                'assets':{'SPY':{'file':'SPY.json','sha256':snapshot['sha256'],
                    'file_sha256':sha256((tmp_path/'SPY.json').read_bytes()).hexdigest()}},'failures':{}}
    write_new(tmp_path/'manifest.json', manifest)
    return tmp_path


def test_run_replay_offline_atomic_bundle(frozen_bundle, monkeypatch, capsys):
    from forecasting import study
    import yfinance
    monkeypatch.setattr(yfinance, 'download', lambda *a, **k: pytest.fail('network access'))
    monkeypatch.setattr(study, 'git_identity', lambda: {'commit':'test','status':'dirty'})
    study.main(['run',str(frozen_bundle)])
    result=frozen_bundle/'results'
    assert len(list(result.glob('*.png')))==3
    assert not list(frozen_bundle.glob('.results-*'))
    assert b'\r\n' not in (result/'results.json').read_bytes()
    saved=json.loads((result/'results.json').read_text())
    saved['SPY']['report']['environment']['matplotlib']='other-version'
    (result/'results.json').write_text(json.dumps(saved),encoding='utf-8')
    study.main(['replay',str(frozen_bundle)])
    assert 'Environment differences' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        study.main(['run',str(frozen_bundle)])
    text=(result/'report.md').read_text(encoding='utf-8')
    assert text.count('|Period|n|Model|')==5
    assert 'Few blocks' in text


@pytest.mark.parametrize('ticker,file', [('SPY','../SPY.json'),('../SPY','SPY.json'),('OTHER','OTHER.json')])
def test_manifest_rejects_asset_paths(frozen_bundle,ticker,file):
    manifest=json.loads((frozen_bundle/'manifest.json').read_text())
    entry=manifest['assets'].pop('SPY'); entry['file']=file
    manifest['assets'][ticker]=entry
    (frozen_bundle/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identity or path'):
        compute(frozen_bundle)


def test_manifest_rejects_swapped_ticker(frozen_bundle):
    manifest=json.loads((frozen_bundle/'manifest.json').read_text())
    manifest['assets']['AAPL']=manifest['assets'].pop('SPY')
    manifest['assets']['AAPL']['file']='AAPL.json'
    (frozen_bundle/'SPY.json').rename(frozen_bundle/'AAPL.json')
    (frozen_bundle/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='ticker'):
        compute(frozen_bundle)


def test_failed_publication_has_no_partial_results(frozen_bundle,monkeypatch):
    from forecasting import study
    monkeypatch.setattr(study,'git_identity',lambda:{})
    def fail(*args):
        raise RuntimeError('figure failure')
    monkeypatch.setattr(study,'figures',fail)
    with pytest.raises(RuntimeError,match='figure failure'):
        study.main(['run',str(frozen_bundle)])
    assert not (frozen_bundle/'results').exists()
    assert not list(frozen_bundle.glob('.results-*'))


def test_legacy_figure_not_overwritten(frozen_bundle):
    from forecasting.study import main
    artifact=frozen_bundle/'SPY-volatility.png'
    artifact.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        main(['run',str(frozen_bundle)])
    assert artifact.read_bytes()==b'original'
