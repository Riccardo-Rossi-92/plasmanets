function [a, E_grid] = PIC_getAcc(x, PIC, Gx, Lx)
% PIC_GETACC  Particle acceleration from the self-consistent electric field.
%
%   One full field solve: CIC deposition of the particle positions x onto the
%   grid, direct solution of the periodic Poisson equation with the Laplacian
%   Lx, the field from the gradient Gx, and CIC gather back to the particles.
%
%   x       : N x 1 particle positions in [0, L)
%   PIC     : configuration and geometry structure
%   Gx, Lx  : periodic gradient and Laplacian matrices, from PIC_operators
%   a       : N x 1 particle acceleration, a = -E
%   E_grid  : Nx x 1 electric field on the grid

Geo = PIC.geo;
N  = PIC.config.N; Nx = PIC.config.Nx;
dx = Geo.dx; L = Geo.Ln;

%% -------------------------------
%  CIC charge deposition
%% -------------------------------

% Cell index
j = floor(x/dx) + 1;
jp1 = j + 1;

% Periodic BC
jp1(jp1 > Nx) = 1;

% CIC weights
xj = (j-1) * dx;
weight_j   = (xj + dx - x) / dx;
weight_jp1 = (x - xj) / dx;

% Deposit density on grid
n = accumarray(j,   weight_j,   [Nx 1], @sum, 0);
n = n + accumarray(jp1, weight_jp1, [Nx 1], @sum, 0);

% Normalize density
n = n * (1 * L) / (N * dx);

%% -------------------------------
%  Solve Poisson equation
%  ∂xx phi = n - n0
%% -------------------------------

rhs = n - 1;
rhs = rhs - mean(rhs);
phi_grid = Lx \ rhs;

%% -------------------------------
%  Electric field on grid
%% -------------------------------

E_grid = -Gx * phi_grid;

%% -------------------------------
%  Gather field to particles
%% -------------------------------

E = weight_j   .* E_grid(j) + ...
    weight_jp1 .* E_grid(jp1);

%% -------------------------------
%  Acceleration
%% -------------------------------

a = -E;

end