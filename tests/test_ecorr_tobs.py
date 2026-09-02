"""Tests for scaling the ECORR basis by integration time."""

from pathlib import Path

import numpy as np
import pytest

import discovery as ds
from discovery import signals as s
from discovery.models import mpta, ppta


DATA = Path(__file__).resolve().parent.parent / "data"
BUILDERS = [s.makegp_ecorr,
            lambda psr, **kw: s.makegp_ecorr_legendre(psr, nmodes=3, **kw),
            s.makegp_quadratic_ecorr_legendre,
            lambda psr, **kw: s.makegp_ecorr_legendre_correlated(psr, nmodes=3, **kw)]


@pytest.fixture
def psr():
    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    p = ds.Pulsar.read_feather(f)
    if 'tobs' not in p.flags:
        # the bundled fixture predates the flag; give it a spread to scale against
        rng = np.random.default_rng(0)
        p.flags['tobs'] = rng.choice(['240', '480', '960'], size=len(p.toas))
    return p


def tobs(psr):
    return np.asarray(psr.flags['tobs']).astype(float)


# --- the weights --------------------------------------------------------------

def test_the_weight_is_the_square_root_of_the_integration_time_ratio(psr):
    w = s.ecorr_tobs_weights(psr, tobs_ref=1800.0)
    assert np.allclose(w, np.sqrt(1800.0 / tobs(psr)))


def test_a_missing_flag_raises(psr):
    del psr.flags['tobs']
    with pytest.raises(KeyError, match="no 'tobs' flag"):
        s.ecorr_tobs_weights(psr)


@pytest.mark.parametrize("bad", ['0', '-240', 'nan'])
def test_a_nonpositive_or_nonfinite_integration_time_raises(psr, bad):
    v = np.asarray(psr.flags['tobs']).astype('U'); v[3] = bad
    psr.flags['tobs'] = v
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        s.ecorr_tobs_weights(psr)


def test_a_nonpositive_reference_raises(psr):
    with pytest.raises(ValueError, match="tobs_ref must be positive"):
        s.ecorr_tobs_weights(psr, tobs_ref=0.0)


def test_a_very_short_integration_warns_but_still_returns(psr, capsys):
    """It carries a large weight, which is a data question, not a modelling one."""
    v = np.asarray(psr.flags['tobs']).astype('U')
    v[5] = str(float(np.median(tobs(psr))) / 100.0)
    psr.flags['tobs'] = v

    w = s.ecorr_tobs_weights(psr)
    assert np.isfinite(w).all()
    assert 'below a tenth of the median' in capsys.readouterr().out


# --- the bases ----------------------------------------------------------------

@pytest.mark.parametrize("build", BUILDERS)
def test_scaling_off_is_the_default_and_changes_nothing(psr, build):
    assert np.array_equal(np.asarray(build(psr).F),
                          np.asarray(build(psr, tobs_scale=False).F))


@pytest.mark.parametrize("build", BUILDERS)
def test_scaling_weights_the_rows_and_keeps_the_shape(psr, build):
    off = np.asarray(build(psr).F)
    on = np.asarray(build(psr, tobs_scale=True).F)
    assert on.shape == off.shape
    assert np.allclose(on, off * np.sqrt(1800.0 / tobs(psr))[:, None])


def test_the_implied_jitter_variance_goes_as_one_over_the_integration_time(psr):
    """A TOA of half the length carries twice the ECORR variance."""
    F = np.asarray(s.makegp_ecorr(psr, tobs_scale=True).F)
    t = tobs(psr)
    lo, hi = int(np.argmin(t)), int(np.argmax(t))
    assert np.isclose((F[lo]**2).sum() / (F[hi]**2).sum(), t[hi] / t[lo], rtol=1e-10)


def test_the_reference_only_rescales_the_amplitude(psr):
    """tobs_ref sets what log10_ecorr means; it does not change the model shape."""
    a = np.asarray(s.makegp_ecorr(psr, tobs_scale=True, tobs_ref=1800.0).F)
    b = np.asarray(s.makegp_ecorr(psr, tobs_scale=True, tobs_ref=3600.0).F)
    assert np.allclose(b, a * np.sqrt(2.0))


# --- the model wiring ---------------------------------------------------------

def test_mpta_threads_the_switch_and_defaults_it_off(psr):
    import inspect

    sig = inspect.signature(mpta.single_pulsar_noise).parameters
    assert sig['ecorr_tobs_scale'].default is False
    assert 'ecorr_tobs_ref' not in sig

    kw = dict(red=False, dm=False, chrom=False, sw=False, chrom_poly=False,
              ecorr_nmodes=3, background=False)
    off = mpta.single_pulsar_noise(psr, **kw, return_components=True)[1]
    on = mpta.single_pulsar_noise(psr, ecorr_tobs_scale=True, **kw, return_components=True)[1]

    def ecorr_basis(comps):
        return [np.asarray(c.F) for c in comps
                if getattr(c, 'gpname', None) and 'ecorr' in str(c.gpname)][0]

    assert np.allclose(ecorr_basis(on),
                       ecorr_basis(off) * np.sqrt(1800.0 / tobs(psr))[:, None])


def test_ppta_threads_the_switch_and_defaults_it_off(psr):
    import inspect

    assert inspect.signature(ppta.single_pulsar_noise).parameters[
        'ecorr_tobs_scale'].default is False
    assert inspect.signature(ppta.makegp_ecorr_ppta).parameters[
        'tobs_scale'].default is False

    off = ppta.makegp_ecorr_ppta(psr, nmodes=2)
    on = ppta.makegp_ecorr_ppta(psr, nmodes=2, tobs_scale=True)
    assert len(on) == len(off)

    w = np.sqrt(1800.0 / tobs(psr))[:, None]
    for a, b in zip(off, on):
        assert np.allclose(np.asarray(b.F), np.asarray(a.F) * w)
