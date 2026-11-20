# CDF-based PINN for the Vlasov–Poisson system

Cumulative-moment (CDF-based) physics-informed neural networks for the
Vlasov–Poisson system, forward and inverse. No velocity grid is introduced at
any stage.

| folder | problem |
|---|---|
| `01_forward_two_stream/` | forward two-stream instability; 20 sequential time windows |
| `02_forward_landau/` | forward Landau damping, perturbative form; 10 windows, warm start |
| `03_inverse_two_stream/` | phase-space reconstruction from macroscopic observables |

Run the `main_*.m` of a case folder from inside that folder. In
`03_inverse_two_stream/`, `main_inverse_n_E.m` uses density and electric field
only and `main_inverse_n_E_u_T.m` also uses momentum and energy; the
observation window width is the variable `DT` at the top of each script.

Requires MATLAB with the Deep Learning Toolbox; set `Config.GPU = 1` for GPU
execution (Parallel Computing Toolbox). See the top-level
[README](../README.md) for the full description and the license.
