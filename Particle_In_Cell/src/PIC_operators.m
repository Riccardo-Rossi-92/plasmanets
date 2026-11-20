function [Gx,Lx] = PIC_operators(Geo)
% PIC_OPERATORS  Periodic finite-difference operators on the x grid.
%
%   Returns the second-order central gradient Gx and Laplacian Lx as sparse
%   Nx-by-Nx matrices, with the corner entries set for periodicity.  Built
%   once and reused at every time step.


%%%%%%%%%%%%% Gradient
dx = Geo.dx;
Nx = Geo.Nx;

e  = ones(Nx,1);

% Gradient matrix (central difference, periodic)
diags = [-1 1];
vals  = [-e e];

Gx = spdiags(vals, diags, Nx, Nx);

% Periodic BC
Gx(1, Nx)   = -1;
Gx(Nx, 1)   =  1;

Gx = Gx / (2*dx);

%%%%%%%%%%%%% Laplacian

e  = ones(Nx,1);

% Laplacian matrix (second central difference, periodic)
diags = [-1 0 1];
vals  = [e -2*e e];

Lx = spdiags(vals, diags, Nx, Nx);

% Periodic BC
Lx(1, Nx) = 1;
Lx(Nx, 1) = 1;

Lx = Lx / dx^2;
end