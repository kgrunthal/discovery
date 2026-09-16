"""Tests for the parallactic-angle-locked delay GP and the angle it is built on."""

import os
from pathlib import Path

import numpy as np
import pytest

import discovery as ds
from discovery import signals as s
from discovery import likelihood as dl


DATA = Path(__file__).resolve().parent.parent / "data"

# the bundled fixture carries no telescope column, so the site is named explicitly,
# which also exercises the site= path. Any site in OBSERVATORY_ITRF will do: these test
# the geometry, not this pulsar's observing history.
SITE = 'meerkat'


@pytest.fixture(scope="module")
def psr():
    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    return ds.Pulsar.read_feather(f)


# --- the angle -------------------------------------------------------------------

def test_gmst_matches_the_defining_value_at_j2000():
    """GMST at J2000.0 is 18h 41m 50.5482s, i.e. 280.46061837 degrees."""
    got = np.rad2deg(s.greenwich_sidereal_angle(51544.5))
    assert abs(got - 280.46061837) < 1e-6


def test_gmst_advances_by_one_sidereal_day():
    """A mean sidereal day is 86164.0905 s, not 86400."""
    a = s.greenwich_sidereal_angle(58000.0)
    b = s.greenwich_sidereal_angle(58000.0 + 86164.0905 / 86400.0)
    assert abs((np.rad2deg(b - a) + 180.0) % 360.0 - 180.0) < 1e-4


def test_parallactic_angle_vanishes_at_transit(psr):
    """At zero hour angle the source, the zenith and the pole are on one great circle.

    The angle is then 0 or pi, according to whether the source transits south or north
    of the zenith, so it is sin(psi) that must vanish and not psi itself.
    """
    lat, lon = s.observatory_location(SITE)
    # solve for the MJDs at which H = 0 by stepping the sidereal rate
    mjd = 58000.0
    for _ in range(60):
        H = s.greenwich_sidereal_angle(mjd) + lon - psr.phi
        mjd -= float(np.arctan2(np.sin(H), np.cos(H))) / (2 * np.pi) * (86164.0905 / 86400.0)

    class AtTransit:
        name, phi, theta = psr.name, psr.phi, psr.theta
        stoas = np.array([mjd * 86400.0])

    assert abs(float(np.sin(s.parallactic_angle(AtTransit, site=SITE)[0]))) < 1e-5


def test_parallactic_angle_uses_site_arrival_times_not_barycentric(psr):
    """psr.toas carry the Roemer delay, which is degrees of hour angle."""
    psi = s.parallactic_angle(psr, site=SITE)

    class Barycentred:
        name, phi, theta = psr.name, psr.phi, psr.theta
        stoas = psr.toas

    other = s.parallactic_angle(Barycentred, site=SITE)
    shift = np.max(np.abs((np.rad2deg(psi - other) + 180.0) % 360.0 - 180.0))
    assert shift > 0.1, "swapping stoas for toas must change the angle appreciably"


def test_parallactic_angle_agrees_with_astropy(psr):
    """Against an independent apparent-sidereal-time implementation."""
    astropy = pytest.importorskip("astropy")
    from astropy.time import Time
    from astropy.coordinates import EarthLocation
    import astropy.units as u

    lat, lon = s.observatory_location(SITE)
    loc = EarthLocation(lat=np.rad2deg(lat) * u.deg, lon=np.rad2deg(lon) * u.deg)
    mjd = np.asarray(psr.stoas) / 86400.0
    t = Time(mjd, format="mjd", scale="utc", location=loc)
    H = (t.sidereal_time("apparent").radian + 0.0) - psr.phi
    dec = 0.5 * np.pi - psr.theta
    ref = np.arctan2(np.sin(H) * np.cos(lat),
                     np.sin(lat) * np.cos(dec) - np.cos(lat) * np.sin(dec) * np.cos(H))

    d = (np.rad2deg(s.parallactic_angle(psr, site=SITE) - ref) + 180.0) % 360.0 - 180.0
    assert np.max(np.abs(d)) < 0.05


def test_observatory_lookup_by_code_and_by_explicit_coordinates():
    assert np.allclose(np.rad2deg(s.observatory_location('pks')),
                       [-32.998406, 148.263510], atol=1e-5)
    assert np.allclose(np.rad2deg(s.observatory_location('meerkat')),
                       [-30.711056, 21.443889], atol=1e-5)
    assert np.allclose(np.rad2deg(s.observatory_location((-30.0, 21.0))), [-30.0, 21.0])


@pytest.mark.skipif(not os.path.exists(os.path.join(os.environ.get('TEMPO2', ''),
                                                    'observatory/observatories.dat')),
                    reason="needs $TEMPO2 for sites outside OBSERVATORY_ITRF")
def test_sites_outside_the_table_come_from_tempo2_by_code_or_name():
    assert np.allclose(s.observatory_location('PARKES'), s.observatory_location('pks'))
    assert np.allclose(np.rad2deg(s.observatory_location('gbt')),
                       [38.433130, -79.839843], atol=1e-5)


def test_unknown_site_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="no coordinates"):
        s.observatory_location('not_a_telescope')


def test_no_site_and_no_telescope_column_raises(psr):
    with pytest.raises(ValueError, match="nowhere to stand"):
        s.parallactic_angle(psr)


def test_telescope_column_is_read_back_from_a_feather(psr, tmp_path):
    """Pulsar.optional_columns must survive a round trip, and stay optional."""
    out = tmp_path / "with.feather"
    psr.telescope = np.array([SITE] * len(psr.toas))
    psr.save_feather(str(out))
    assert np.all(ds.Pulsar.read_feather(str(out)).telescope == SITE)

    del psr.telescope
    plain = tmp_path / "without.feather"
    psr.save_feather(str(plain))          # must not raise on a Pulsar lacking it
    assert not hasattr(ds.Pulsar.read_feather(str(plain)), 'telescope')


# --- the basis -------------------------------------------------------------------

def test_column_count_and_disjoint_bin_support(psr):
    gp = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE,
                                project_tm=False)
    F = np.asarray(gp.F)
    nbin = len(gp.pa_bins)
    assert F.shape[1] == 2 * nbin

    lab = np.asarray(psr.flags['chan']).astype(str)
    order = sorted(set(lab.tolist()), key=int)
    for j, value in enumerate(order):
        outside = lab != value
        assert np.all(F[outside, j] == 0.0)
        assert np.all(F[outside, j + nbin] == 0.0)


def test_one_scale_governs_the_whole_basis(psr):
    gp = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE)
    assert gp.Phi.params == [f'{psr.name}_pa_gp_log10_sigma']
    assert gp.pa_harmonic == 2


def test_a_nonpositive_harmonic_raises(psr):
    with pytest.raises(ValueError, match="positive integer"):
        s.makegp_pa_quadrature(psr, harmonic=0, bin_flag='chan', site=SITE)


def test_a_missing_bin_flag_raises_rather_than_guessing_the_bins(psr):
    """The bins are the receiver channelisation; inferring them would be a guess."""
    with pytest.raises(ValueError, match="needs bin_flag"):
        s.makegp_pa_quadrature(psr, bin_flag=None, site=SITE)
    with pytest.raises(KeyError, match="has no flag"):
        s.makegp_pa_quadrature(psr, bin_flag='not_a_flag', site=SITE)


def test_the_implied_delay_covariance_is_coherent_in_pa_within_a_bin(psr):
    """sigma**2 F F^T must be sigma**2 cos(m (psi_i - psi_j)) inside a bin, zero across.

    That is the whole model: within a channel the delay is a sinusoid in the
    parallactic angle whose amplitude AND phase are free, and channels are independent.
    """
    gp = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE,
                                project_tm=False)
    F = np.asarray(gp.F)
    psi = s.parallactic_angle(psr, site=SITE)
    lab = np.asarray(psr.flags['chan']).astype(str)

    idx = np.arange(len(psi))[:400]
    K = F[idx] @ F[idx].T
    same = lab[idx][:, None] == lab[idx][None, :]
    expect = np.where(same, np.cos(2 * (psi[idx][:, None] - psi[idx][None, :])), 0.0)
    assert np.allclose(K, expect, atol=1e-12)


def test_sigma_is_the_per_quadrature_delay(psr):
    """Every TOA gets variance sigma**2, so a channel amplitude has E[A**2] = 2 sigma**2."""
    gp = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE,
                                project_tm=False)
    F = np.asarray(gp.F)
    assert np.allclose(np.sum(F**2, axis=1), 1.0, atol=1e-12)


def test_prior_box_is_registered_for_every_scale(psr):
    from discovery import prior
    s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE)
    import re
    key = f'{re.escape(psr.name)}_pa_gp_log10_sigma'
    assert prior.priordict_standard[key] == [-10.0, -6.0]


def test_projection_keeps_the_columns_rather_than_dropping_them(psr):
    """The column count must not depend on what the projection annihilates.

    An annihilated direction integrates back to its prior; dropping it would change
    the prior instead.
    """
    raw = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE,
                                 project_tm=False)
    proj = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE,
                                  project_tm=True)
    assert np.asarray(proj.F).shape == np.asarray(raw.F).shape


def test_projecting_an_improperly_marginalised_basis_is_a_no_op(psr):
    """Removing the fd span changes nothing when that span carries an improper prior."""
    fd = s.makegp_fd_piecewise(psr, spacing='flag', kind='constant', bin_flag='chan',
                               name='fd')

    def logl(project):
        pa = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE,
                                    project=fd if project else None)
        m = dl.PulsarLikelihood([psr.residuals, s.makenoise_measurement(psr, {}),
                                 s.makegp_timing(psr, svd=True), fd, pa])
        p = {k: (1.0 if k.endswith('efac') else -7.5 if 'pa_gp' in k else -8.0)
             for k in m.logL.params}
        return float(m.logL(p))

    assert abs(logl(True) - logl(False)) < 1e-6


def test_injected_pa_delay_is_recovered_at_the_right_scale(psr):
    """Inject a known per-channel sinusoid in PA and scan the marginal likelihood."""
    rng = np.random.default_rng(20260827)
    psi = s.parallactic_angle(psr, site=SITE)
    lab = np.asarray(psr.flags['chan']).astype(str)
    order = sorted(set(lab.tolist()), key=int)

    # Scaled to the fixture rather than fixed, so the test cannot quietly become
    # untestable: two per-channel amplitudes are measured from ~n_chan TOAs each, so the
    # amplitude error is err / sqrt(n_chan / 2) and this injects twice that.
    err = np.asarray(psr.toaerrs)
    per_bin = np.median([int((lab == v).sum()) for v in order])
    sigma_true = float(2.0 * np.median(err) / np.sqrt(0.5 * per_bin))

    delay = np.zeros(len(psi))
    for value in order:
        m = lab == value
        a, b = rng.normal(scale=sigma_true, size=2)
        delay[m] = a * np.sin(2 * psi[m]) + b * np.cos(2 * psi[m])

    res = delay + rng.normal(scale=err)

    pa = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE)
    m = dl.PulsarLikelihood([res, s.makenoise_measurement(psr, {}),
                             s.makegp_timing(psr, svd=True), pa])
    grid = np.linspace(-10.0, -6.0, 41)
    key = f'{psr.name}_pa_gp_log10_sigma'
    base = {k: (1.0 if k.endswith('efac') else -8.5) for k in m.logL.params}
    curve = np.array([float(m.logL({**base, key: g})) for g in grid])

    best = grid[int(np.argmax(curve))]
    assert abs(best - np.log10(sigma_true)) < 0.3, (best, np.log10(sigma_true))
    assert curve.max() - curve[0] > 20.0, "an injected term must beat the no-term end"


# --- threading through common_noise -----------------------------------------------

def _fake_chain(psr, pa=False, pa_common=False, boost=None, pa_variant=None):
    """Minimal stage-1 chain: red noise, per-backend efac, and optionally the PA scale,
    with the phase too for the common-phase variant."""
    import pandas as pd

    cols = [f'{psr.name}_red_noise_log10_A', f'{psr.name}_red_noise_gamma']
    cols += [f'{psr.name}_{be}_efac'
             for be in sorted(set(np.asarray(psr.backend_flags).tolist()))]
    if pa or pa_common:
        cols += ([f'{psr.name}_pa_gp_log10_sigma'] if pa_variant is None else
                 [f'{psr.name}_pa_gp_log10_sigma_X', f'{psr.name}_pa_gp_log10_sigma_Y'] +
                 ([f'{psr.name}_pa_gp_rho_XY'] if pa_variant == 'full' else []))
    if pa_common:
        cols += [f'{psr.name}_pa_gp_phase']
    if boost == 'iso':
        cols += [f'{psr.name}_pa_boost_log10_sigma']
    elif boost in ('diag', 'full'):
        cols += [f'{psr.name}_pa_boost_log10_sigma_X', f'{psr.name}_pa_boost_log10_sigma_Y']
        if boost == 'full':
            cols += [f'{psr.name}_pa_boost_rho_XY']

    df = pd.DataFrame({c: np.linspace(-8.0, -7.0, 8) for c in cols})
    df.attrs['noisedict'] = {}
    return df


@pytest.fixture(scope="module")
def two_psrs():
    """Two pulsars carrying a telescope column, which the mpta path requires.

    common_noise does not take a site: it relies on the column, which every MPTA and
    PPTA feather has and these bundled fixtures do not.
    """
    files = [DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather",
             DATA / "v1p1_de440_pint_bipm2019-B1855+09.feather"]
    if not all(f.exists() for f in files):
        pytest.skip("pulsar data fixtures missing")

    psrs = [ds.Pulsar.read_feather(f) for f in files]
    for psr in psrs:
        psr.telescope = np.array([SITE] * len(psr.toas))
    return psrs


def test_common_noise_switches_the_pa_gp_on_per_pulsar_from_the_chains(two_psrs):
    """Presence comes from the chain, as every other component does."""
    from discovery.models import mpta

    a, b = two_psrs
    m = mpta.common_noise(two_psrs, [_fake_chain(a, pa=True), _fake_chain(b)],
                          fd=False, pa_bin_flag='chan', noise_point='median')

    assert sorted(p for p in m.logL.params if 'pa_gp' in p) == \
        [f'{a.name}_pa_gp_log10_sigma']


def test_common_noise_leaves_the_pa_gp_out_where_no_chain_carries_it(two_psrs):
    from discovery.models import mpta

    a, b = two_psrs
    m = mpta.common_noise(two_psrs, [_fake_chain(a), _fake_chain(b)],
                          fd=False, pa_bin_flag='chan', noise_point='median')
    assert not [p for p in m.logL.params if 'pa_gp' in p]


def test_common_noise_warns_about_the_settings_the_chain_cannot_carry(two_psrs, capsys):
    """The bins and the projection change the basis, not the parameters."""
    from discovery.models import mpta

    a, b = two_psrs
    mpta.common_noise(two_psrs, [_fake_chain(a, pa=True), _fake_chain(b)],
                      fd=False, pa_bin_flag='chan', noise_point='median')

    out = capsys.readouterr().out
    assert 'a disagreement cannot be reported' in out
    assert "bin_flag='chan'" in out
    assert 'pa_project_fd=True' in out
    assert f'1 of {len(two_psrs)} pulsar(s) carry' in out


def test_commongp_falls_back_rather_than_stacking_a_variable_core(two_psrs, capsys):
    """The PA GP samples its scale, so it leaves the per-pulsar core variable."""
    from discovery.models import mpta

    a, b = two_psrs
    mpta.common_noise(two_psrs, [_fake_chain(a, pa=True), _fake_chain(b)],
                      fd=False, pa_bin_flag='chan',
                      use_commongp=True, fix_chrom_alpha=True, noise_point='median')

    out = capsys.readouterr().out
    assert 'parallactic-angle GP, which is not stackable' in out
    assert 'Falling back to the GlobalLikelihood path' in out


# --- one phase across channels ----------------------------------------------------

def _phase_key(psr):
    return f'{psr.name}_pa_gp_phase'


def _params_at(m, psr, theta, log10_sigma):
    return {k: (1.0 if k.endswith('efac') else theta if k == _phase_key(psr)
                else log10_sigma if 'pa_gp' in k else -8.0) for k in m.logL.params}


def test_common_phase_is_the_free_basis_collapsed_onto_one_phase(psr):
    """Column c is 1[chan=c] sin(m psi + theta) = F_sin cos(theta) + F_cos sin(theta)."""
    free = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE, project_tm=False)
    Ff = np.asarray(free.F)
    n = Ff.shape[1] // 2
    gp = s.makegp_pa_common_phase(psr, bin_flag='chan', site=SITE, project_tm=False)
    assert callable(gp.F)
    for theta in (-1.2, 0.0, 0.7):
        Fc = np.asarray(gp.F({_phase_key(psr): theta}))
        assert Fc.shape == (Ff.shape[0], n)
        assert np.allclose(Fc, Ff[:, :n] * np.cos(theta) + Ff[:, n:] * np.sin(theta), atol=1e-12)


def test_common_phase_coefficient_is_the_signed_channel_amplitude(psr):
    """Each column is sin(2 psi + theta) on exactly one channel and zero elsewhere."""
    gp = s.makegp_pa_common_phase(psr, bin_flag='chan', site=SITE, project_tm=False)
    psi = s.parallactic_angle(psr, site=SITE)
    lab = np.asarray(psr.flags['chan']).astype(str)
    F = np.asarray(gp.F({_phase_key(psr): 0.4}))
    for j in range(F.shape[1]):
        on = F[:, j] != 0.0
        assert len(set(lab[on].tolist())) == 1
        assert np.allclose(F[on, j], np.sin(2 * psi[on] + 0.4), atol=1e-12)


def test_common_phase_prior_boxes_are_registered(psr):
    import re
    from discovery import prior
    s.makegp_pa_common_phase(psr, bin_flag='chan', site=SITE)
    assert prior.priordict_standard[f'{re.escape(psr.name)}_pa_gp_log10_sigma'] == [-10.0, -6.0]
    assert prior.priordict_standard[f'{re.escape(psr.name)}_pa_gp_phase'] == \
        [-0.5 * float(np.pi), 0.5 * float(np.pi)]


def test_common_phase_projection_holds_at_every_phase(psr):
    """The blocks are projected once; the combination must stay orthogonal to the span."""
    gp = s.makegp_pa_common_phase(psr, bin_flag='chan', site=SITE, project_tm=True)
    Q = np.linalg.qr(s.normalise_tm_basis(psr))[0]
    for theta in (-1.0, 0.3, 1.4):
        F = np.asarray(gp.F({_phase_key(psr): theta}))
        assert np.abs(Q.T @ F).max() < 1e-9


def test_common_phase_fixed_phase_gives_a_constant_basis_and_no_parameter(psr):
    var = s.makegp_pa_common_phase(psr, bin_flag='chan', site=SITE)
    fix = s.makegp_pa_common_phase(psr, bin_flag='chan', site=SITE, phase=0.6)
    assert not callable(fix.F)
    assert np.allclose(np.asarray(fix.F), np.asarray(var.F({_phase_key(psr): 0.6})), atol=1e-12)
    m = dl.PulsarLikelihood([psr.residuals, s.makenoise_measurement(psr, {}),
                             s.makegp_timing(psr, svd=True), fix])
    assert _phase_key(psr) not in m.logL.params
    assert f'{psr.name}_pa_gp_log10_sigma' in m.logL.params


def _inject_common_phase(psr, theta_true, rng):
    """Signed per-channel amplitudes on one phase, at twice the per-channel amplitude error."""
    psi = s.parallactic_angle(psr, site=SITE)
    lab = np.asarray(psr.flags['chan']).astype(str)
    order = sorted(set(lab.tolist()), key=int)
    err = np.asarray(psr.toaerrs)
    per_bin = np.median([int((lab == v).sum()) for v in order])
    sigma_true = float(2.0 * np.median(err) / np.sqrt(0.5 * per_bin))
    delay = np.zeros(len(psi))
    for v in order:
        m = lab == v
        delay[m] = rng.normal(scale=sigma_true) * np.sin(2 * psi[m] + theta_true)
    return delay + rng.normal(scale=err), sigma_true


def test_common_phase_is_pi_periodic_and_the_phase_matters(psr):
    """A signed amplitude makes theta and theta + pi one model; theta + pi/2 is not."""
    res, sigma_true = _inject_common_phase(psr, 0.4, np.random.default_rng(3))
    gp = s.makegp_pa_common_phase(psr, bin_flag='chan', site=SITE)
    m = dl.PulsarLikelihood([res, s.makenoise_measurement(psr, {}),
                             s.makegp_timing(psr, svd=True), gp])
    ls = np.log10(sigma_true)
    l0 = float(m.logL(_params_at(m, psr, 0.4, ls)))
    l1 = float(m.logL(_params_at(m, psr, 0.4 + np.pi, ls)))
    l2 = float(m.logL(_params_at(m, psr, 0.4 + np.pi / 2, ls)))
    assert abs(l0 - l1) < 1e-6 * max(1.0, abs(l0))
    assert abs(l0 - l2) > 1.0


def test_common_phase_injection_is_covered_by_the_phase_profile_and_recovers_the_scale(psr):
    """The truth must lie inside the 2-nat interval of the theta profile, whatever the
    fixture's parallactic-angle coverage resolves, and sigma must come back at scale."""
    theta_true = 0.5
    res, sigma_true = _inject_common_phase(psr, theta_true, np.random.default_rng(20260912))
    gp = s.makegp_pa_common_phase(psr, bin_flag='chan', site=SITE)
    m = dl.PulsarLikelihood([res, s.makenoise_measurement(psr, {}),
                             s.makegp_timing(psr, svd=True), gp])
    ls = np.log10(sigma_true)

    thetas = np.linspace(-np.pi / 2, np.pi / 2, 73)[:-1]
    curve = np.array([float(m.logL(_params_at(m, psr, t, ls))) for t in thetas])
    curve -= curve.max()
    assert curve.min() < -20.0, "the phase must matter somewhere on the half-turn"
    at_truth = float(m.logL(_params_at(m, psr, theta_true, ls))) - (curve.max() +
               float(m.logL(_params_at(m, psr, thetas[int(np.argmax(curve))], ls))) - curve.max())
    assert at_truth > -2.0, at_truth

    grid = np.linspace(-10.0, -6.0, 41)
    curve2 = np.array([float(m.logL(_params_at(m, psr, theta_true, g))) for g in grid])
    assert abs(grid[int(np.argmax(curve2))] - ls) < 0.3
    assert curve2.max() - curve2[0] > 20.0


def test_single_pulsar_noise_pa_phase_selects_the_basis(two_psrs):
    from discovery.models import mpta

    a = two_psrs[0]
    free = mpta.single_pulsar_noise(a, fftint=False, pa_gp=True, pa_phase='free')
    common = mpta.single_pulsar_noise(a, fftint=False, pa_gp=True, pa_phase='common')
    fixed = mpta.single_pulsar_noise(a, fftint=False, pa_gp=True, pa_phase=0.21)
    assert _phase_key(a) not in free.logL.params
    assert _phase_key(a) in common.logL.params
    assert _phase_key(a) not in fixed.logL.params
    assert all(f'{a.name}_pa_gp_log10_sigma' in m.logL.params for m in (free, common, fixed))
    with pytest.raises(ValueError, match="pa_phase must be"):
        mpta.single_pulsar_noise(a, fftint=False, pa_gp=True, pa_phase='both')


def test_common_noise_reads_the_common_phase_variant_from_the_chains(two_psrs):
    """A chain carrying pa_gp_phase gets the one-phase basis; one without keeps the free one."""
    from discovery.models import mpta

    a, b = two_psrs
    m = mpta.common_noise(two_psrs, [_fake_chain(a, pa_common=True), _fake_chain(b, pa=True)],
                          fd=False, pa_bin_flag='chan', noise_point='median')
    assert sorted(p for p in m.logL.params if 'pa_gp' in p) == sorted([
        f'{a.name}_pa_gp_log10_sigma', _phase_key(a), f'{b.name}_pa_gp_log10_sigma'])


def test_common_noise_pa_phase_override_applies_to_every_carrier(two_psrs):
    """A fixed-phase stage-1 run is indistinguishable from the free one, so it is declared."""
    from discovery.models import mpta

    a, b = two_psrs
    m = mpta.common_noise(two_psrs, [_fake_chain(a, pa=True), _fake_chain(b)],
                          fd=False, pa_bin_flag='chan', noise_point='median', pa_phase=0.21)
    pa = sorted(p for p in m.logL.params if 'pa_gp' in p)
    assert pa == [f'{a.name}_pa_gp_log10_sigma']
    m2 = mpta.common_noise(two_psrs, [_fake_chain(a, pa=True), _fake_chain(b)],
                           fd=False, pa_bin_flag='chan', noise_point='median', pa_phase='common')
    assert _phase_key(a) in m2.logL.params


# --- boost-unit basis with per-channel susceptibilities -----------------------------

def _labels(psr):
    return sorted(set(np.asarray(psr.flags['chan']).astype(str).tolist()), key=int)


def _z(psr, zQ=0.0, zU=1.0, seed=None):
    if seed is None:
        return {lab: (zQ, zU) for lab in _labels(psr)}
    rng = np.random.default_rng(seed)
    return {lab: tuple(rng.normal(scale=1e-5, size=2)) for lab in _labels(psr)}


def test_boost_with_unit_u_susceptibility_is_the_quadrature_basis(psr):
    """zQ = 0, zU = 1 s, chi0 = 0, h = +1 must give exactly [sin 2psi | cos 2psi]."""
    for project_tm in (False, True):
        quad = s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE, project_tm=project_tm)
        boost = s.makegp_pa_boost(psr, _z(psr), bin_flag='chan', site=SITE, project_tm=project_tm)
        assert np.allclose(np.asarray(boost.F), np.asarray(quad.F), rtol=0.0, atol=1e-14)


def test_boost_columns_follow_the_pib_rotation(psr):
    """dX = zQ cos 2h psi + zU sin 2h psi and dY = -zQ sin 2h psi + zU cos 2h psi, per channel."""
    z = _z(psr, seed=5)
    psi = s.parallactic_angle(psr, site=SITE)
    lab = np.asarray(psr.flags['chan']).astype(str)
    for h in (1, -1):
        gp = s.makegp_pa_boost(psr, z, bin_flag='chan', site=SITE, hand=h, project_tm=False)
        F = np.asarray(gp.F); n = F.shape[1] // 2
        for j, v in enumerate(gp.pa_labels):
            m = lab == v; zQ, zU = z[v]
            assert np.allclose(F[m, j], zQ * np.cos(2 * h * psi[m]) + zU * np.sin(2 * h * psi[m]), atol=1e-16)
            assert np.allclose(F[m, n + j], -zQ * np.sin(2 * h * psi[m]) + zU * np.cos(2 * h * psi[m]), atol=1e-16)
            assert np.all(F[~m, j] == 0.0) and np.all(F[~m, n + j] == 0.0)


def test_boost_chi0_offsets_the_angle(psr):
    """A fixed chi0 equals evaluating the h = +1 rotation at psi + chi0."""
    z = _z(psr, seed=6)
    psi = s.parallactic_angle(psr, site=SITE)
    lab = np.asarray(psr.flags['chan']).astype(str)
    gp = s.makegp_pa_boost(psr, z, bin_flag='chan', site=SITE, chi0=0.3, project_tm=False)
    F = np.asarray(gp.F); n = F.shape[1] // 2
    j = 0; v = gp.pa_labels[0]; m = lab == v; zQ, zU = z[v]
    assert np.allclose(F[m, j], zQ * np.cos(2 * (psi[m] + 0.3)) + zU * np.sin(2 * (psi[m] + 0.3)), atol=1e-16)
    assert np.allclose(F[m, n + j], -zQ * np.sin(2 * (psi[m] + 0.3)) + zU * np.cos(2 * (psi[m] + 0.3)), atol=1e-16)


def test_boost_sampled_chi0_is_callable_and_matches_the_fixed_basis(psr):
    z = _z(psr, seed=7)
    var = s.makegp_pa_boost(psr, z, bin_flag='chan', site=SITE, chi0='sample')
    assert callable(var.F)
    for c in (-0.5, 0.0, 0.4):
        fix = s.makegp_pa_boost(psr, z, bin_flag='chan', site=SITE, chi0=c)
        assert np.allclose(np.asarray(var.F({f'{psr.name}_pa_boost_chi0': c})), np.asarray(fix.F), atol=1e-16)


def test_boost_prior_is_kron_of_the_two_by_two_covariance(psr):
    """Phi = I_nchan (x) D R D in the [X | Y] column order, for the three variants."""
    z = _z(psr, seed=8)
    for variant in ('iso', 'diag', 'full'):
        gp = s.makegp_pa_boost(psr, z, bin_flag='chan', site=SITE, variant=variant, project_tm=False)
        n = np.asarray(gp.F).shape[1] // 2
        base = f'{psr.name}_pa_boost'
        pars = ({f'{base}_log10_sigma': -2.0} if variant == 'iso' else
                {f'{base}_log10_sigma_X': -2.0, f'{base}_log10_sigma_Y': -3.0})
        if variant == 'full':
            pars[f'{base}_rho_XY'] = 0.4
        Phi = np.asarray(gp.Phi.getN(pars))
        sX = 1e-2; sY = 1e-2 if variant == 'iso' else 1e-3; r = 0.4 if variant == 'full' else 0.0
        M = np.array([[sX * sX, r * sX * sY], [r * sX * sY, sY * sY]])
        assert np.allclose(Phi, np.kron(M, np.eye(n)), rtol=1e-12)
        assert sorted(gp.Phi.getN.params if hasattr(gp.Phi.getN, 'params') else gp.Phi.params) == sorted(pars)


def test_boost_prior_boxes_are_in_boost_units(psr):
    import re
    from discovery import prior
    s.makegp_pa_boost(psr, _z(psr), bin_flag='chan', site=SITE, variant='full', chi0='sample')
    e = re.escape(psr.name)
    assert prior.priordict_standard[f'{e}_pa_boost_log10_sigma_X'] == [-6.0, -1.3]
    assert prior.priordict_standard[f'{e}_pa_boost_log10_sigma_Y'] == [-6.0, -1.3]
    assert prior.priordict_standard[f'{e}_pa_boost_rho_XY'] == [-1.0, 1.0]
    assert prior.priordict_standard[f'{e}_pa_boost_chi0'] == [-0.25 * float(np.pi), 0.25 * float(np.pi)]
    s.makegp_pa_boost(psr, _z(psr), bin_flag='chan', site=SITE, variant='iso')
    assert prior.priordict_standard[f'{e}_pa_boost_log10_sigma'] == [-6.0, -1.3]


def test_boost_iso_likelihood_is_invariant_under_handedness(psr):
    """Both hands span the same columns; an isotropic prior makes the two models one."""
    z = _z(psr, seed=9)
    rng = np.random.default_rng(1)
    res = np.asarray(psr.residuals) + rng.normal(scale=np.asarray(psr.toaerrs))
    out = []
    for h in (1, -1):
        gp = s.makegp_pa_boost(psr, z, bin_flag='chan', site=SITE, hand=h, variant='iso')
        m = dl.PulsarLikelihood([res, s.makenoise_measurement(psr, {}), s.makegp_timing(psr, svd=True), gp])
        pr = {k: (1.0 if k.endswith('efac') else -2.5 if 'pa_boost' in k else -8.0) for k in m.logL.params}
        out.append(float(m.logL(pr)))
    assert abs(out[0] - out[1]) < 1e-6 * max(1.0, abs(out[0]))


def test_boost_anisotropic_likelihood_is_not_hand_invariant_when_zU_is_nonzero(psr):
    """Flipping h maps b to its reflection about the susceptibility direction, which is
    bY -> -bY only for zU = 0; with zU != 0 an anisotropic prior tells the hands apart."""
    z = _z(psr, seed=10)
    rng = np.random.default_rng(2)
    res = np.asarray(psr.residuals) + rng.normal(scale=np.asarray(psr.toaerrs))
    out = []
    for h in (1, -1):
        gp = s.makegp_pa_boost(psr, z, bin_flag='chan', site=SITE, hand=h, variant='diag')
        m = dl.PulsarLikelihood([res, s.makenoise_measurement(psr, {}), s.makegp_timing(psr, svd=True), gp])
        pr = {k: (1.0 if k.endswith('efac') else -2.0 if k.endswith('sigma_X') else -4.0 if k.endswith('sigma_Y')
                  else -8.0) for k in m.logL.params}
        out.append(float(m.logL(pr)))
    assert abs(out[0] - out[1]) > 1e-3


def test_boost_era_split_doubles_the_blocks_with_disjoint_rows(psr):
    z = _z(psr, seed=11)
    t = np.asarray(getattr(psr, 'stoas', psr.toas)) / 86400.0
    cut = float(np.median(t))
    one = s.makegp_pa_boost(psr, z, bin_flag='chan', site=SITE, project_tm=False)
    two = s.makegp_pa_boost(psr, z, bin_flag='chan', site=SITE, project_tm=False, era_split=cut)
    F1, F2 = np.asarray(one.F), np.asarray(two.F); n = F1.shape[1]
    assert F2.shape == (F1.shape[0], 2 * n)
    early = t < cut
    assert np.all(F2[~early, :n] == 0.0) and np.all(F2[early, n:] == 0.0)
    assert np.array_equal(F2[early, :n], F1[early]) and np.array_equal(F2[~early, n:], F1[~early])
    Phi = np.asarray(two.Phi.getN({f'{psr.name}_pa_boost_log10_sigma_X': -2.0,
                                   f'{psr.name}_pa_boost_log10_sigma_Y': -2.0,
                                   f'{psr.name}_pa_boost_rho_XY': 0.0}))
    assert Phi.shape == (2 * n, 2 * n)
    with pytest.raises(ValueError, match="leaves an era with no TOAs"):
        s.makegp_pa_boost(psr, z, bin_flag='chan', site=SITE, era_split=t.min() - 1.0)


def test_boost_rejects_bad_susceptibilities(psr):
    z = _z(psr); z.pop(next(iter(z)))
    with pytest.raises(KeyError, match="no susceptibility"):
        s.makegp_pa_boost(psr, z, bin_flag='chan', site=SITE)
    with pytest.raises(ValueError, match="z must give"):
        s.makegp_pa_boost(psr, np.zeros((3, 2)), bin_flag='chan', site=SITE)
    with pytest.raises(ValueError, match="variant must be"):
        s.makegp_pa_boost(psr, _z(psr), bin_flag='chan', site=SITE, variant='axial')
    with pytest.raises(ValueError, match="hand must be"):
        s.makegp_pa_boost(psr, _z(psr), bin_flag='chan', site=SITE, hand=2)


def test_single_pulsar_noise_builds_the_boost_gp_and_refuses_both(two_psrs):
    from discovery.models import mpta

    a = two_psrs[0]
    m = mpta.single_pulsar_noise(a, fftint=False, pa_boost=True, pa_boost_z=_z(a, seed=12))
    assert f'{a.name}_pa_boost_log10_sigma_X' in m.logL.params
    assert f'{a.name}_pa_boost_rho_XY' in m.logL.params
    assert not any('pa_gp' in p for p in m.logL.params)
    with pytest.raises(ValueError, match="span the same columns"):
        mpta.single_pulsar_noise(a, fftint=False, pa_gp=True, pa_boost=True, pa_boost_z=_z(a))
    with pytest.raises(ValueError, match="needs pa_boost_z"):
        mpta.single_pulsar_noise(a, fftint=False, pa_boost=True)


def test_common_noise_reads_the_boost_variant_from_the_chains(two_psrs):
    from discovery.models import mpta

    a, b = two_psrs
    m = mpta.common_noise(two_psrs, [_fake_chain(a, boost='diag'), _fake_chain(b, boost='iso')],
                          fd=False, pa_bin_flag='chan', noise_point='median',
                          pa_boost_z={a.name: _z(a, seed=13), b.name: _z(b, seed=14)})
    got = sorted(p for p in m.logL.params if 'pa_boost' in p)
    assert got == sorted([f'{a.name}_pa_boost_log10_sigma_X', f'{a.name}_pa_boost_log10_sigma_Y',
                          f'{b.name}_pa_boost_log10_sigma'])
    with pytest.raises(ValueError, match="no susceptibilities"):
        mpta.common_noise(two_psrs, [_fake_chain(a, boost='full'), _fake_chain(b)],
                          fd=False, pa_bin_flag='chan', noise_point='median')


# --- anisotropy in the free-phase pair ---------------------------------------------

def _pa(psr, variant='iso', **kw):
    return s.makegp_pa_quadrature(psr, bin_flag='chan', site=SITE, variant=variant, **kw)


def _K(gp, params, idx):
    """Implied delay covariance F Phi F^T on a subset of TOAs, for either Phi shape."""
    F = np.asarray(gp.F)[idx]
    Phi = np.asarray(gp.Phi.getN(params))
    return F @ (Phi @ F.T) if Phi.ndim == 2 else F @ (Phi[:, None] * F.T)


def _scales(psr, sx, sy, rho=None):
    out = {f'{psr.name}_pa_gp_log10_sigma_X': np.log10(sx),
           f'{psr.name}_pa_gp_log10_sigma_Y': np.log10(sy)}
    if rho is not None:
        out[f'{psr.name}_pa_gp_rho_XY'] = rho
    return out


def test_quadrature_defaults_to_one_scale(psr):
    gp = _pa(psr)
    assert gp.pa_variant == 'iso'
    assert gp.Phi.getN.params == [f'{psr.name}_pa_gp_log10_sigma']


def test_quadrature_anisotropic_parameter_names_and_boxes(psr):
    import re
    from discovery import prior
    e = re.escape(psr.name)
    assert _pa(psr, 'diag').Phi.getN.params == [f'{psr.name}_pa_gp_log10_sigma_X',
                                                f'{psr.name}_pa_gp_log10_sigma_Y']
    assert prior.priordict_standard[f'{e}_pa_gp_log10_sigma_X'] == [-10.0, -6.0]
    assert prior.priordict_standard[f'{e}_pa_gp_log10_sigma_Y'] == [-10.0, -6.0]
    assert _pa(psr, 'full').Phi.getN.params == [f'{psr.name}_pa_gp_log10_sigma_X',
                                                f'{psr.name}_pa_gp_log10_sigma_Y',
                                                f'{psr.name}_pa_gp_rho_XY']
    assert prior.priordict_standard[f'{e}_pa_gp_rho_XY'] == [-1.0, 1.0]


def test_quadrature_phi_shapes_split_the_blocks(psr):
    """diag stays diagonal, one scale per quadrature block; full is M (x) I over channels."""
    n = np.asarray(_pa(psr).F).shape[1] // 2
    d = np.asarray(_pa(psr, 'diag').Phi.getN(_scales(psr, 1e-7, 1e-9)))
    assert d.shape == (2 * n,)
    assert np.allclose(d[:n], 1e-14) and np.allclose(d[n:], 1e-18)
    f = np.asarray(_pa(psr, 'full').Phi.getN(_scales(psr, 1e-7, 1e-9, 0.3)))
    M = np.array([[1e-14, 0.3 * 1e-7 * 1e-9], [0.3 * 1e-7 * 1e-9, 1e-18]])
    assert np.allclose(f, np.kron(M, np.eye(n)), rtol=1e-12)


def test_quadrature_variants_nest_at_equal_scales(psr):
    """diag with one scale, and full with rho = 0, are the isotropic model exactly."""
    idx = np.arange(0, len(psr.toas), 23)[:300]
    sig = 3e-8
    Ki = _K(_pa(psr, project_tm=False), {f'{psr.name}_pa_gp_log10_sigma': np.log10(sig)}, idx)
    Kd = _K(_pa(psr, 'diag', project_tm=False), _scales(psr, sig, sig), idx)
    Kf = _K(_pa(psr, 'full', project_tm=False), _scales(psr, sig, sig, 0.0), idx)
    assert np.allclose(Kd, Ki, rtol=0.0, atol=1e-30)
    assert np.allclose(Kf, Ki, rtol=0.0, atol=1e-30)


def _shifted(psr, variant, chi):
    """The quadrature basis with the origin of psi moved by chi, from the shipped boost
    builder: with z = (0, 1 s) its columns are [sin 2(psi + chi) | cos 2(psi + chi)]."""
    z = {lab: (0.0, 1.0) for lab in sorted(set(np.asarray(psr.flags['chan']).astype(str).tolist()), key=int)}
    return s.makegp_pa_boost(psr, z, bin_flag='chan', site=SITE, variant=variant,
                             chi0=chi, project_tm=False)


def _bscales(psr, sx, sy, rho=None):
    out = {f'{psr.name}_pa_boost_log10_sigma_X': np.log10(sx),
           f'{psr.name}_pa_boost_log10_sigma_Y': np.log10(sy)}
    if rho is not None:
        out[f'{psr.name}_pa_boost_rho_XY'] = rho
    return out


def test_iso_is_invariant_to_the_psi_origin_and_diag_is_not(psr):
    """psi = 0 is transit, an arbitrary origin: only a rotation-covariant prior is a
    statement about the instrument rather than about the source's culmination."""
    idx = np.arange(0, len(psr.toas), 23)[:300]
    sx, sy, chi = 3e-8, 1e-8, 0.35
    Ki0 = _K(_shifted(psr, 'iso', 0.0), {f'{psr.name}_pa_boost_log10_sigma': np.log10(sx)}, idx)
    Kic = _K(_shifted(psr, 'iso', chi), {f'{psr.name}_pa_boost_log10_sigma': np.log10(sx)}, idx)
    assert np.abs(Kic - Ki0).max() / np.abs(Ki0).max() < 1e-12

    Kd0 = _K(_shifted(psr, 'diag', 0.0), _bscales(psr, sx, sy), idx)
    Kdc = _K(_shifted(psr, 'diag', chi), _bscales(psr, sx, sy), idx)
    assert np.abs(Kdc - Kd0).max() / np.abs(Kd0).max() > 0.1


def test_full_is_closed_under_a_shift_of_the_psi_origin(psr):
    """diag at origin chi is full at origin 0 with M -> R(2 chi) M R(2 chi)^T."""
    idx = np.arange(0, len(psr.toas), 23)[:300]
    sx, sy, chi = 3e-8, 1e-8, 0.35
    Kdc = _K(_shifted(psr, 'diag', chi), _bscales(psr, sx, sy), idx)

    c2, s2 = np.cos(2 * chi), np.sin(2 * chi)
    R = np.array([[c2, -s2], [s2, c2]])
    M = R @ np.diag([sx**2, sy**2]) @ R.T
    sxp, syp = np.sqrt(M[0, 0]), np.sqrt(M[1, 1])
    Kf = _K(_pa(psr, 'full', project_tm=False), _scales(psr, sxp, syp, M[0, 1] / (sxp * syp)), idx)
    assert np.abs(Kf - Kdc).max() / np.abs(Kdc).max() < 1e-12


def test_full_at_unit_correlation_is_the_common_phase_model(psr):
    """rho = +-1 with equal scales collapses each channel pair onto sin(2 psi +- pi/4)."""
    idx = np.arange(0, len(psr.toas), 23)[:300]
    sig = 3e-8
    cp = s.makegp_pa_common_phase(psr, bin_flag='chan', site=SITE, project_tm=False)
    for rho, theta in ((1.0, np.pi / 4), (-1.0, -np.pi / 4)):
        Kf = _K(_pa(psr, 'full', project_tm=False), _scales(psr, sig, sig, rho), idx)
        Fc = np.asarray(cp.F({f'{psr.name}_pa_gp_phase': theta}))[idx]
        Kc = (2 * sig**2) * (Fc @ Fc.T)
        assert np.abs(Kf - Kc).max() / np.abs(Kc).max() < 1e-12


def test_quadrature_rejects_an_unknown_variant(psr):
    with pytest.raises(ValueError, match="variant must be 'iso', 'diag' or 'full'"):
        _pa(psr, 'axial')


def test_single_pulsar_noise_pa_variant_selects_the_prior(two_psrs):
    from discovery.models import mpta

    a = two_psrs[0]
    iso = mpta.single_pulsar_noise(a, fftint=False, pa_gp=True)
    full = mpta.single_pulsar_noise(a, fftint=False, pa_gp=True, pa_variant='full')
    assert f'{a.name}_pa_gp_log10_sigma' in iso.logL.params
    assert f'{a.name}_pa_gp_rho_XY' in full.logL.params
    assert f'{a.name}_pa_gp_log10_sigma' not in full.logL.params
    with pytest.raises(ValueError, match="applies to the free-phase model"):
        mpta.single_pulsar_noise(a, fftint=False, pa_gp=True, pa_phase='common', pa_variant='diag')


def test_common_noise_reads_the_pa_variant_from_the_chains(two_psrs):
    from discovery.models import mpta

    a, b = two_psrs
    m = mpta.common_noise(two_psrs, [_fake_chain(a, pa=True, pa_variant='full'), _fake_chain(b, pa=True)],
                          fd=False, pa_bin_flag='chan', noise_point='median')
    assert sorted(p for p in m.logL.params if 'pa_gp' in p) == sorted([
        f'{a.name}_pa_gp_log10_sigma_X', f'{a.name}_pa_gp_log10_sigma_Y', f'{a.name}_pa_gp_rho_XY',
        f'{b.name}_pa_gp_log10_sigma'])
