"""PPTA common noise: component detection from a stage-1 chain, and the white-noise path."""

from pathlib import Path

import numpy as np
import pytest

import discovery as ds


DATA = Path(__file__).resolve().parent.parent / "data"
PSR = "J0437-4715"


@pytest.fixture
def ppta():
    from discovery.models import ppta
    return ppta


def cols(psr, **kw):
    """Stage-1 chain columns for a PPTA-shaped run.

    ecorr_modes: Legendre modes on the 'global' block, 0 for no global block
    correlated:  add the mode-correlation parameters
    backends:    per-system plain ECORR labels
    """
    c = [f'{psr}_red_noise_log10_A', f'{psr}_red_noise_gamma',
         f'{psr}_dm_gp_log10_A', f'{psr}_dm_gp_gamma']
    for be in kw.get('white', ['CASPSR_40CM', 'UWL_sbB']):
        c += [f'{psr}_{be}_efac', f'{psr}_{be}_log10_tnequad']
    n = kw.get('ecorr_modes', 3)
    if n:
        c.append(f'{psr}_global_log10_ecorr')
        c += [f'{psr}_global_log10_ecorr_k{k}' for k in range(1, n)]
        if kw.get('correlated', True):
            c += [f'{psr}_global_ecorr_corr_k{i}k{j}'
                  for i in range(1, n) for j in range(i)]
    for be in kw.get('backends', []):
        c.append(f'{psr}_{be}_log10_ecorr')
    for g in kw.get('groups', []):
        c += [f'{psr}_group_noise_{g}_log10_A', f'{psr}_group_noise_{g}_gamma']
    c += kw.get('extra', [])
    return c


# --- detection ---------------------------------------------------------------------

def test_legendre_on_the_global_block_and_plain_ecorr_per_system(ppta):
    """PPTA's shape: a correlated Legendre stack on 'global', plain ECORR per system."""
    d = ppta.detect_ppta_components(PSR, cols(PSR, ecorr_modes=3, correlated=True,
                                              backends=['UWL_sbB', 'PDFB_20CM']))
    assert d['ecorr'] and d['ecorr_nmodes'] == 3 and d['ecorr_correlated']
    assert d['ecorr_per_backend']
    assert d['ecorr_dict'] == {PSR: ['PDFB_20CM', 'UWL_sbB']}


def test_the_global_block_is_omitted_when_the_chain_has_none(ppta):
    """nmodes=None omits it; a per-system-only chain has no _k columns either."""
    d = ppta.detect_ppta_components(PSR, cols(PSR, ecorr_modes=0, backends=['UWL_sbB']))
    assert d['ecorr_nmodes'] is None
    assert d['ecorr_dict'] == {PSR: ['UWL_sbB']}


def test_a_single_mode_global_block_is_not_mistaken_for_none(ppta):
    d = ppta.detect_ppta_components(PSR, cols(PSR, ecorr_modes=1, correlated=False))
    assert d['ecorr_nmodes'] == 1
    assert not d['ecorr_correlated']
    assert d['ecorr_dict'] is None


def test_group_noise_labels_come_from_the_chain(ppta):
    d = ppta.detect_ppta_components(PSR, cols(PSR, groups=['CASPSR_40CM', 'UWL_sbH']))
    assert d['groups'] == ['CASPSR_40CM', 'UWL_sbH']
    assert d['group_dict'] == {PSR: ['CASPSR_40CM', 'UWL_sbH']}


def test_no_group_noise_leaves_the_dict_unset(ppta):
    d = ppta.detect_ppta_components(PSR, cols(PSR))
    assert d['groups'] == [] and d['group_dict'] is None


@pytest.mark.parametrize("extra,kernel,powerlaw", [
    ([f'{PSR}_sw_gp_log10_sigma', f'{PSR}_sw_gp_log10_ell'], 'se', False),
    ([f'{PSR}_sw_gp_log10_sigma', f'{PSR}_sw_gp_log10_ell',
      f'{PSR}_sw_gp_log10_Gamma', f'{PSR}_sw_gp_log10_p'], 'qp', False),
    ([f'{PSR}_sw_gp_log10_A', f'{PSR}_sw_gp_gamma'], 'se', True),
])
def test_the_solar_wind_kernel_is_read_from_its_parameters(ppta, extra, kernel, powerlaw):
    d = ppta.detect_ppta_components(PSR, cols(PSR, extra=extra))
    assert d['sw'] and d['sw_kernel'] == kernel and d['sw_powerlaw'] is powerlaw


def test_no_solar_wind_parameters_means_no_solar_wind(ppta):
    assert not ppta.detect_ppta_components(PSR, cols(PSR))['sw']


def test_a_turnover_is_detected_per_component(ppta):
    d = ppta.detect_ppta_components(PSR, cols(PSR, extra=[f'{PSR}_red_noise_log10_fc']))
    assert d['turnover'] == ('red',)
    assert ppta.detect_ppta_components(PSR, cols(PSR))['turnover'] == ()


def test_another_pulsars_columns_do_not_leak_in(ppta):
    """A neighbour's backends must not join this pulsar's."""
    mixed = cols(PSR, backends=['UWL_sbB']) + cols('J1909-3744', backends=['UWL_sbG'])
    d = ppta.detect_ppta_components(PSR, mixed)
    assert d['ecorr_dict'] == {PSR: ['UWL_sbB']}


# --- the white-noise path ----------------------------------------------------------

@pytest.fixture(scope="module")
def psr():
    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    return ds.Pulsar.read_feather(f)


def _chain(psr, white=True):
    import pandas as pd

    backends = sorted(set(np.asarray(psr.backend_flags).tolist()))
    c = [f'{psr.name}_red_noise_log10_A', f'{psr.name}_red_noise_gamma']
    c += [f'{psr.name}_global_log10_ecorr']
    if white:
        for be in backends:
            c += [f'{psr.name}_{be}_efac', f'{psr.name}_{be}_log10_tnequad']

    df = pd.DataFrame({k: np.linspace(-8.0, -7.0, 8) if 'log10' in k
                       else np.linspace(0.9, 1.1, 8) for k in c})
    df.attrs['noisedict'] = {}
    return df


def _white(m):
    return sorted(p for p in m.logL.params
                  if p.endswith('_efac') or 'equad' in p or 'ecorr' in p)


def test_white_noise_is_fixed_from_the_chain(ppta, psr):
    m = ppta.common_noise([psr], [_chain(psr)], noise_point='median')
    assert _white(m) == []


def test_white_noise_is_fitted_when_the_chain_does_not_carry_it(ppta, psr):
    m = ppta.common_noise([psr], [_chain(psr, white=False)],
                          noise_point='median')
    assert len(_white(m)) == 2 * len(set(np.asarray(psr.backend_flags).tolist()))


def test_white_selection_reaches_the_per_pulsar_rebuild(ppta, psr, capsys):
    """A split the stage-1 names cannot match is reported, not applied silently."""
    ppta.common_noise([psr], [_chain(psr)], noise_point='median',
                      white_selection='chan')
    assert 'efac parameter(s) this rebuild cannot produce' in capsys.readouterr().out


# --- fix_chrom_alpha ---------------------------------------------------------------

@pytest.mark.parametrize("chrom_poly", [False, True])
def test_a_fixed_chromatic_index_stays_fixed_with_the_polynomial(ppta, psr, chrom_poly):
    """The polynomial GP shares chrom_gp_alpha, so it must be given the same value."""
    m = ppta.single_pulsar_noise(psr, chrom_alpha=6.596, chrom_poly=chrom_poly,
                                 red=False, dm=False, sw=False, ecorr=False)
    assert [p for p in m.logL.params if 'alpha' in p] == []


@pytest.mark.parametrize("chrom_poly", [False, True])
def test_an_unfixed_chromatic_index_is_sampled_either_way(ppta, psr, chrom_poly):
    m = ppta.single_pulsar_noise(psr, chrom_alpha=None, chrom_poly=chrom_poly,
                                 red=False, dm=False, sw=False, ecorr=False)
    assert [p for p in m.logL.params if 'alpha' in p] == [f'{psr.name}_chrom_gp_alpha']


def test_a_fixed_index_makes_the_polynomial_basis_constant(ppta, psr):
    """Paths that stack the per-pulsar GPs need a non-callable design matrix."""
    from discovery import signals as sg
    from discovery.models import mpta

    fixed = sg.makegp_chrom_poly_svd(psr, name='chrom_gp',
                                     noisedict=mpta._chrom_poly_noisedict(psr, 6.596))
    sampled = sg.makegp_chrom_poly_svd(psr, name='chrom_gp',
                                       noisedict=mpta._chrom_poly_noisedict(psr, None))
    assert not callable(fixed.F)
    assert callable(sampled.F)


def test_fix_chrom_alpha_reaches_the_rebuild(ppta, psr):
    """common_noise defaults fix_chrom_alpha=True, which must not be a no-op."""
    df = _chain(psr)
    df[f'{psr.name}_chrom_gp_log10_A'] = np.linspace(-14.0, -13.0, 8)
    df[f'{psr.name}_chrom_gp_gamma'] = np.linspace(2.0, 3.0, 8)
    df[f'{psr.name}_chrom_gp_alpha'] = np.linspace(6.0, 7.0, 8)

    fixed = ppta.common_noise([psr], [df], noise_point='median', fix_chrom_alpha=True)
    free = ppta.common_noise([psr], [df], noise_point='median', fix_chrom_alpha=False)

    assert [p for p in fixed.logL.params if 'alpha' in p] == []
    assert [p for p in free.logL.params if 'alpha' in p] == [f'{psr.name}_chrom_gp_alpha']


# --- PEBBLE and chrom_fref -----------------------------------------------------------

def _has_partials():
    import os
    from discovery import phys_ephem as pe
    return os.path.exists(pe.DEFAULT_PARTIALS)


pebble = pytest.mark.skipif(not _has_partials(), reason="PEBBLE partials file missing")


def _pe(m):
    return sorted(p for p in m.logL.params if 'phys_ephem' in p)


@pebble
def test_pebble_is_off_unless_asked_for(ppta, psr):
    m = ppta.common_noise([psr], [_chain(psr)], noise_point='median')
    assert _pe(m) == []


@pebble
def test_pebble_carries_the_jupiter_and_saturn_orbital_elements(ppta, psr):
    m = ppta.common_noise([psr], [_chain(psr)], noise_point='median',
                          use_phys_ephem=True)
    got = _pe(m)
    assert 'phys_ephem_jupiter_orbit(6)' in got
    assert 'phys_ephem_saturn_orbit(6)' in got
    # common to the array, so the names carry no pulsar prefix
    assert not any(p.startswith(psr.name) for p in got)


@pebble
@pytest.mark.parametrize("switch,dropped", [
    ('phys_ephem_inc_jupiter', 'phys_ephem_jupiter_orbit(6)'),
    ('phys_ephem_inc_saturn', 'phys_ephem_saturn_orbit(6)'),
    ('phys_ephem_inc_jerk', 'phys_ephem_ssb_jerk(3)'),
])
def test_each_pebble_switch_reaches_the_delay(ppta, psr, switch, dropped):
    df = _chain(psr)
    on = _pe(ppta.common_noise([psr], [df], noise_point='median', use_phys_ephem=True))
    off = _pe(ppta.common_noise([psr], [df], noise_point='median', use_phys_ephem=True,
                                **{switch: False}))
    assert dropped in on
    assert set(on) - set(off) == {dropped}


@pebble
def test_mass_bodies_sizes_the_mass_block(ppta, psr):
    df = _chain(psr)
    four = _pe(ppta.common_noise([psr], [df], noise_point='median', use_phys_ephem=True))
    one = _pe(ppta.common_noise([psr], [df], noise_point='median', use_phys_ephem=True,
                                phys_ephem_mass_bodies=('jupiter',)))
    assert 'phys_ephem_mass(4)' in four
    # a length-one vector parameter drops the count suffix
    assert 'phys_ephem_mass' in one


def test_chrom_fref_changes_the_chromatic_basis(ppta, psr):
    """It scales the basis as (fref/nu)**alpha, so two values are two models."""
    kw = dict(chrom_alpha=4.0, red=False, dm=False, sw=False, ecorr=False,
              chrom_poly=False, fftint=False, return_components=True)
    _, a = ppta.single_pulsar_noise(psr, chrom_fref=1400.0, **kw)
    _, b = ppta.single_pulsar_noise(psr, chrom_fref=1100.0, **kw)

    fa = [np.asarray(g.F) for g in a if getattr(g, 'gpname', None) == 'chrom_gp'][0]
    fb = [np.asarray(g.F) for g in b if getattr(g, 'gpname', None) == 'chrom_gp'][0]
    assert fa.shape == fb.shape
    assert not np.allclose(fa, fb)


# --- chromatic events ----------------------------------------------------------------

@pytest.mark.parametrize("cols_,exp,unlabelled", [
    ([f'{PSR}_chrom_exp_1_alpha', f'{PSR}_chrom_exp_1_t0'], True, False),
    ([f'{PSR}_chrom_exp_alpha', f'{PSR}_chrom_exp_t0'], False, True),
    ([f'{PSR}_chrom_exp_1_t0', f'{PSR}_chrom_exp_t0'], True, True),
    ([], False, False),
])
def test_an_indexed_event_does_not_switch_on_the_unlabelled_one(ppta, cols_, exp,
                                                                unlabelled):
    """models_dict events are {psr}_chrom_exp_{i}_*; chrom_exponential adds one
    unindexed event beside them."""
    d = ppta.detect_ppta_components(PSR, cols(PSR) + cols_)
    assert d['chrom_exp'] is exp
    assert d['chrom_exponential'] is unlabelled


@pytest.mark.parametrize("name,key", [
    ('chrom_1yr', 'chrom_annual'),
    ('chrom_gauss', 'chrom_gauss'),
    ('gauss_20cm', 'chrom_gauss_20cm'),
    ('chrom_sphere', 'chrom_sphere'),
    ('chrom_step', 'chrom_step'),
])
def test_the_unindexed_events_are_detected_by_name(ppta, name, key):
    assert ppta.detect_ppta_components(PSR, cols(PSR) + [f'{PSR}_{name}_log10_Amp'])[key]
    assert not ppta.detect_ppta_components(PSR, cols(PSR))[key]


def test_a_20cm_event_is_not_read_as_a_gaussian_one(ppta):
    d = ppta.detect_ppta_components(PSR, cols(PSR) + [f'{PSR}_gauss_20cm_t0'])
    assert d['chrom_gauss_20cm'] and not d['chrom_gauss']
