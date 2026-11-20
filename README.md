# Raman_spec

Post-processing tool that turns an ABINIT **anaddb** output file into a Raman
spectrum: per-mode intensities, polarization-resolved channels, depolarization
ratios, theoretical cross-sections and a Lorentzian-broadened spectrum ready
for plotting.

It reads the text `anaddb.out` produced by a `nlflag > 0` run, extracts the
zone-centre phonon frequencies and their 3x3 Raman susceptibility tensors, and
writes plain-text output. Nothing is plotted for you — the output is columnar
text so you can use whatever plotting tool you prefer.

---

## Requirements

- Python 3.8+
- `numpy`
- `scipy`

```bash
pip install numpy scipy
```

No compiled extensions, no ABINIT installation needed — only the text output
file an anaddb run already produced.

---

## Quick start

```bash
python Raman_spec.py raman.in
```

where `raman.in` is a small plain-text input file:

```
# AlN wurtzite, 532 nm excitation
filename   anaddb.out
temp       300.0
laser_freq 532.0 nm
spread     2.0 cm-1
freq_unit  cm-1
n_freq     400
outname    aln
calctype   1
relative_intensity
```

Add `-v` or `--verbose` for DEBUG-level output and per-mode progress.

---

## Input file reference

One key per line. Blank lines and lines starting with `#` are ignored.
Unrecognised keys are silently skipped.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `filename` | path | *required* | anaddb **text** output to read (not `.nc`) |
| `laser_freq` | value + unit | *required* | Incident photon energy; must be > 0 |
| `temp` | float | `300.0` | Temperature in K, for the Bose–Einstein factor |
| `spread` | value + unit | `0.0 cm-1` | Lorentzian **half**-width. `0` disables broadening and writes per-mode intensities instead of a spectrum |
| `freq_unit` | unit | `cm-1` | Unit of the output frequency axis |
| `n_freq` | int | `1000` | Number of points in the broadened spectrum |
| `min_freq` | value + unit | auto | Lower bound of the spectrum; negative means auto-select |
| `max_freq` | value + unit | auto | Upper bound; negative means auto-select |
| `outname` | string | `raman` | Output filename prefix |
| `calctype` | int | `0` | Calculation type code |
| `relative_intensity` | flag | off | Normalize each channel to its own maximum |
| `keep_file` | flag | off | Overwrite existing output instead of writing a numbered variant |
| `verbose` | flag | off | DEBUG-level logging |

### Units

Frequency units are `Ha` / `hartree`, `Hz` / `hertz`, or `cm-1` / `cm^-1` /
`wavenumber`. A laser is more naturally given as a wavelength, so `nm` and `um`
are also accepted and converted to cm⁻¹ for you:

```
laser_freq 532.0 nm       # equivalent to 18796.99 cm-1
laser_freq 18797.0 cm-1
```

Auto-selected spectrum bounds are taken from the Raman-active modes, padded by
5%. Acoustic and imaginary modes are excluded from that choice so an unstable
structure does not push the axis below zero.

---

## Output files

### `<outname>_spec` — broadened spectrum (when `spread > 0`)

Eight whitespace-separated columns:

```
# freq (cm1) rel Ipowder   rel I(XX)   rel I(XY)   rel I(XZ)   rel I(YY)   rel I(YZ)   rel I(ZZ)
2.25663e+02 5.90474e-04 6.47689e-04 6.28345e-04 2.18786e-05 6.47686e-04 2.18786e-05 2.96600e-05
```

Column 1 is the frequency in the requested unit; columns 2–8 are the powder
average followed by the six Cartesian polarization channels, in the fixed order
`POWDER, XX, XY, XZ, YY, YZ, ZZ`. The header says `abs` or `rel` depending on
`relative_intensity`.

One file is written per q-direction. The transverse (TO) block uses the prefix
as given; each longitudinal (LO) direction gets the q-vector appended, e.g.
`aln_spec_1000_0000_0000`.

### `<outname>_intensity` — per-mode intensities (when `spread = 0`)

Same columns, one row per mode instead of a continuous axis.

### `<outname>_dielectric_freq`

Written when the anaddb run also contained a frequency-dependent dielectric
calculation (`dieflag`).

### `raman.log`

Full run log, UTF-8. Includes each mode's Raman tensor, its invariants, the
depolarization ratio, Raman activity, optical activity ratio, theoretical
cross-section and — when a `spread` was supplied — phonon lifetime and quality
factor, plus a ranked summary of the ten strongest modes.

---

## Physics and conventions

### Tensor invariants

Splitting the Raman tensor into its symmetric part `s = (m + mᵀ)/2` and
antisymmetric part `a = (m − mᵀ)/2`:

```
G0 = (1/3) · Tr(m)²                 isotropic
G1 = Σᵢⱼ aᵢⱼ²                       antisymmetric
G2 = Σᵢⱼ sᵢⱼ² − (1/3) · Tr(m)²      symmetric anisotropic
```

**These match ABINIT's own convention.** anaddb prints a
`Spherical averages : G0= ... G1= ... G2= ...` line under each mode's tensor,
and the values computed here reproduce it, so results are directly comparable
against the source file.

### Intensities

The powder channel is the orientation-averaged Placzek result:

```
I_par  = 2π · pref · (10·G0 + 4·G2)
I_perp = 2π · pref · ( 5·G1 + 3·G2)
I_powder = I_par + I_perp
```

with

```
pref = (1 / 2ω_m) · (ω_L − ω_m)⁴ / c⁴ · (n(ω_m, T) + 1)
```

evaluated in Hartree atomic units, where `ω_m` is the mode frequency, `ω_L` the
incident photon frequency, `ω_L − ω_m` the Stokes-scattered photon and `n` the
Bose–Einstein population. These coefficients reproduce the textbook
`I_∥ ∝ 45a² + 4γ²`, `I_⊥ ∝ 3γ²` with `γ² = (3/2)·G2` and `a² = G0/3`.

The six fixed-polarization channels are `|ê_s · α · ê_i|² · pref` at one
orientation.

> **Normalization caveat.** The `2π` applies to the powder channel only, where
> it comes from the solid-angle integration of the orientational average. The
> fixed-polarization channels have had no such integration performed. The seven
> columns therefore sit on scales differing by that factor even though they are
> written side by side. Compare powder-to-powder and channel-to-channel; do not
> read the ratio of column 2 to column 3 as physical.

### Depolarization ratio

Reported as the orientation-averaged quantity

```
ρ = I_perp / I_par = (5·G1 + 3·G2) / (10·G0 + 4·G2)
```

computed from the tensor invariants directly. `ρ = 0` for a purely isotropic
tensor, `ρ = 3/4` for a symmetric traceless one, and above `3/4` only once the
tensor has an antisymmetric part. It is suppressed (`--`) for modes whose
intensity is negligible against the strongest, because anaddb prints rounding
noise rather than exact zeros for Raman-inactive modes and a ratio formed from
that noise looks deceptively like a measurement.

### Cross-sections

`calculate_theoretical_raman_cross_section` evaluates the differential Stokes
cross-section

```
dσ/dΩ = 2π · (ω_s⁴ / c⁴) · (n + 1) / (2ω_m) · (10·G0 + 5·G1 + 7·G2)
```

in atomic units, converted to Å²/sr. This is the same expression the powder
channel evaluates, so a mode's cross-section and its powder intensity sit on
one consistent scale.

**No empirical prefactor, scaling constant or reference-material calibration is
applied.** The absolute magnitude inherits the approximations of the underlying
DFT Raman tensor and of the dimensionless-polarizability convention anaddb
writes it in. Treat it as a first-principles estimate, not a measured
cross-section.

### Thermodynamic quantities

`get_mode_heat_capacity` and `get_mode_entropy` return the **single harmonic
oscillator (Einstein)** per-mode results in units of k_B — not a Debye model,
which would need a density of states and a Debye temperature that do not appear
here.

### Lineshape

Each mode contributes `width / ((ω − ω_m)² + width²)`, an unnormalized
Lorentzian of **half**-width `spread`, evaluated in Hartree. The constant `1/π`
of a normalized Lorentzian is omitted; since it is common to every mode it
affects only the absolute scale.

`estimate_linewidth_quality` interprets the user-supplied broadening as the
mode's physical damping width to report `Q = ω/Γ` and `τ = 1/(2πΓ)`. That is a
modelling choice, not a computed anharmonic linewidth — these numbers are only
as meaningful as the `spread` you supplied.

---

## Validation

Checked against a real ABINIT run on wurtzite AlN (12 modes, 7 Raman-active):

- **Invariants.** `G0` and `G2` reproduce anaddb's own printed
  `Spherical averages` for every active mode, to the precision of the
  9-decimal tensors in the file.
- **Depolarization ratios.** The totally-symmetric A₁(TO) mode at 601.26 cm⁻¹
  gives ρ = 0.7479, just below 3/4 as its small isotropic component requires;
  every doubly-degenerate E mode (237.54, 643.89, 655.76 cm⁻¹) gives exactly
  0.7500.
- **Frequencies.** 237.54 / 601.26 / 643.89 / 655.76 cm⁻¹ against experimental
  E₂(low) 249, A₁(TO) 611, E₁(TO) 671, E₂(high) 657 cm⁻¹.

`G1` is zero for every mode in that file, as expected for non-resonant Raman
tensors, which are symmetric by construction. The antisymmetric path is
therefore implemented but **not validated against reference data**.

---

## Known limitations

- The antisymmetric (`G1`) path is untested against real data; every anaddb
  tensor encountered so far has been symmetric.
- `calctype` is accepted and stored but currently selects no alternative
  behaviour.
- Spectra are written as text only; no plotting is included.
- Only the anaddb **text** output is read. NetCDF (`.nc`) is rejected with a
  clear error rather than parsed.

---

## Programmatic use

```python
import Raman_spec as R

# Full pipeline, either from an input file ...
config = R.run('raman.in')

# ... or entirely from keyword arguments, with no input file at all.
config = R.run(filename='anaddb.out', temp=300.0,
               laser_freq='532 nm', spread='2 cm-1',
               freq_unit='cm1', n_freq=400, outname='aln')

# Individual pieces are usable on their own.
G0, G1, G2 = R.get_g012(tensor)         # invariants of one 3x3 tensor
rho = R.get_powder_depolarization(tensor)
sigma = R.calculate_theoretical_raman_cross_section(
    tensor, mode_frequency_cm1=601.26,
    laser_frequency_cm1=18797.0, temperature_k=300.0)
```

Pass *either* a path *or* keyword arguments — when a path is given the
input file is the source of truth and most keyword overrides are ignored.

---

## License

MIT — see [LICENSE](LICENSE).

## Citing

If this tool contributes to published work, please cite ABINIT and anaddb for
the underlying Raman susceptibilities, alongside this repository.
