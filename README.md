# Vlasov–Poisson: phase-space reconstruction from time-resolved fluid moments

Code accompanying two companion manuscripts by
R. Rossi, P. Gaudio, M. Gelfusa and A. Murari:

- **Reconstructing Phase-Space Distributions with Time-Resolved Fluid Moments
  and Physics-Informed Neural Networks** — the inverse problem and the
  methodology. *Submitted.*
- a companion **Letter** on how a finite observation window lifts the moment
  degeneracy. *In preparation.*

Affiliations: Department of Industrial Engineering, University of Rome
"Tor Vergata"; Consorzio RFX, Padova; ISTP–CNR, Padova.

## Contents

```
Grid_Based/                        Fourier-spectral Vlasov–Poisson solver (1D1V)
Particle_In_Cell/                  electrostatic PIC code (1D1V)
Physics_Informed_Neural_Networks/  CDF-based PINN, forward and inverse problems
```

**`Grid_Based/`** — reference solver used for the degeneracy and
separation study. Fourier-spectral in both `x` and `v`, so each advection is
an exact shift applied as a phase factor; time integration by a fourth-order
Yoshida composition of Strang splittings. Defaults: `Nx = 32`, `Nv = 2048`,
`v ∈ [-10, 10]`, `dt = 2e-3`. Each script is self-contained and runs unchanged
in MATLAB and in GNU Octave (~1 min).

| script | purpose |
|---|---|
| `Numerical_Simulation_Vlasov.m` | base run, integration to `k·t = 2` |
| `Numerical_Simulation_Vlasov_allMoments.m` | same, extended to `k·t = 20` |
| `Numerical_Simulation_Vlasov_parametric_k.m` | scan over `k = 0.25, 1, 5` |

**`Particle_In_Cell/`** — in-house electrostatic PIC, periodic over one
wavelength `L = 2π/k`. 32 cells, 10⁶ particles per cell; quiet start by exact
inversion of the CDF at Hammersley quantiles; CIC deposition and gather;
second-order periodic Poisson solved directly; leapfrog kick–drift–kick.
`dt = 1e-2`, output every `1e-1`, up to `t = 20 ωpe⁻¹`. No velocity grid enters
the dynamics. Select the initial condition with the `Case` variable in
`main_pic.m` (`"maxwell"`, `"two-beam"`, `"kappa"`); helper functions in `src/`.

**`Physics_Informed_Neural_Networks/`** — PINN formulated in cumulative
moments, so no velocity grid is introduced at any stage. Each case folder holds
its `main_*.m`, a `src/` with the network, the loss gradients and the domain
setup, the reference PIC data under `PIC/`, and an empty `results/`.

| folder | problem |
|---|---|
| `01_forward_two_stream/` | forward two-stream instability; 20 sequential time windows, one network each |
| `02_forward_landau/` | forward Landau damping, perturbative form; 10 windows with warm start |
| `03_inverse_two_stream/` | phase-space reconstruction over a single observation window |

In `03_inverse_two_stream/`, `main_inverse_n_E.m` uses density and electric
field only, `main_inverse_n_E_u_T.m` adds momentum and energy. The observation
window width is the variable `DT` at the top of each script.

## Requirements

- `Grid_Based/`: MATLAB or GNU Octave, no toolboxes.
- `Particle_In_Cell/`: MATLAB.
- `Physics_Informed_Neural_Networks/`: MATLAB R2021a or later with the Deep
  Learning Toolbox (`dlarray`, `dlfeval`, `adamupdate`). GPU execution is
  optional: set `Config.GPU = 1` at the top of the main script, which requires
  the Parallel Computing Toolbox and a CUDA-capable device.

Paths inside the scripts use Windows separators (`src\`, `PIC\...`); on Linux
and macOS replace them with `/`.

## Running

`cd` into a case folder and run its `main_*.m` — the scripts use relative
paths. PINN runs write `.mat` snapshots into `results/`.

The `.mat` files under each `PIC/` folder are the reference particle-in-cell
runs (~57 MB in total) and were produced with the code in
`Particle_In_Cell/`.

## Citation

Please cite the accompanying papers. Full references will be added here on
publication.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Riccardo Rossi — r.rossi@ing.uniroma2.it
