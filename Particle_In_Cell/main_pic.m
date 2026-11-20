%% MAIN_PIC  Electrostatic particle-in-cell simulation of a 1D1V plasma
%
%   Periodic domain of a single wavelength, L = 2*pi/kwave.  Particles are
%   loaded by exact inversion of the CDF of the chosen initial distribution
%   at the quantiles of a Hammersley sequence (quiet start: no initial
%   statistical noise).  Charge is deposited and the field gathered with CIC
%   weighting, the periodic Poisson equation is solved directly with a
%   second-order central Laplacian, and particles are advanced with a
%   leapfrog kick-drift-kick scheme.
%
%   Set the initial condition with the variable Case below:
%       "maxwell"   Maxwellian at T = 1
%       "two-beam"  two symmetric beams, v_b = sqrt(chi*T)
%       "kappa"     kappa distribution, kappa = 3.5
%
%   No velocity grid enters the dynamics: PIC.config.Nv is used only to
%   histogram the particles for diagnostics, so the run is free of the
%   recurrence of a grid-based Vlasov solver.
%
%   Output: the Results structure, one entry every PIC.time.dt_results, each
%   holding the time, the field E, the histogram f and its moments n, u, K.
%   Helper functions in src/.

clear; clc;
addpath src\

%% Configuration

kwave = 1.00;

% pic discretisation configuration
PIC.config.N_cell = 1e6; % particles per cell
PIC.config.Nx = 32;   % number of cell on x (used for simulation)
PIC.config.Nv = 501;    % number of cell on v (used only for plot)
PIC.config.N = PIC.config.N_cell*PIC.config.Nx; % total number of particles

% pic geometry configuration
PIC.geo.range_xn = [0 2*pi./kwave]; % x boundary
PIC.geo.range_vn = [-6 6]; % v boundary

% pic time configuration
PIC.time.t_start = 0;
PIC.time.t_end = 20;
PIC.time.dt_sim = 1e-2; % simulation dt
PIC.time.dt_results = 1e-1; % results dt

%%

% Case = "maxwell"; % "maxwell", "two-beam"
% Case = "two-beam";
Case = "kappa";

if Case == "maxwell"

    PIC = PIC_geometry_1D(PIC);

    x = PIC.geo.xgn;
    v = PIC.geo.vgn;
    dv = PIC.geo.dv;

    T = 1;
    v_th = sqrt(T);
    p = 0.1;

    A = 1 + p*cos(kwave*x);

    % reference - Maxwell
    PIC.IC.f = 1/((2*pi).^0.5*v_th)*exp(-(v./v_th).^2/2).*A;

    PIC.Mom.n_id = sum(PIC.IC.f.*dv,1);
    PIC.Mom.u_id = sum(PIC.IC.f.*v.*dv,1);
    PIC.Mom.T_id = sum(PIC.IC.f.*v.^2.*dv,1);

    [x, v] = sample_particles_from_f(PIC.IC.f, x(1,:), v(:,1), PIC.config.N);

    class = x*0;

elseif Case == "two-beam"

    PIC = PIC_geometry_1D(PIC);

    x = PIC.geo.xgn;
    v = PIC.geo.vgn;
    dv = PIC.geo.dv;

    chi = 0.79;
    T = 1;

    v_b = sqrt(chi.*T);
    v_s = sqrt(T-v_b.^2);

    v_th = 0.2;

    p = 0.1;

    A = 1 + p*cos(kwave*x);

    PIC.IC.f = 1/((2*pi).^0.5.*v_s)*(exp(-((v-v_b)./v_s).^2/2) + exp(-((v+v_b)./v_s).^2/2))/2;
    PIC.IC.f =  PIC.IC.f.*A;

    PIC.Mom.n_id = sum(PIC.IC.f.*dv,1);
    PIC.Mom.u_id = sum(PIC.IC.f.*v.*dv,1);
    PIC.Mom.T_id = sum(PIC.IC.f.*v.^2.*dv,1);


    [x, v] = sample_particles_from_f(PIC.IC.f, x(1,:), v(:,1), PIC.config.N);

    class = x*0;

elseif Case == "kappa"

    PIC = PIC_geometry_1D(PIC);

    x = PIC.geo.xgn;
    v = PIC.geo.vgn;
    dv = PIC.geo.dv;

    % v_th = 1.03;
    v_th = 1.0093;

    p = 0.1;

    A = 1 + p*cos(kwave*x);

    % reference - Maxwell
    kap = 3.5;   c = gamma(kap)/(gamma(kap-0.5)*sqrt(pi*(2*kap-3)));
    PIC.IC.f = A .* (c/v_th).*(1 + (v/v_th).^2/(2*kap-3)).^(-kap);

    PIC.Mom.n_id = sum(PIC.IC.f.*dv,1);
    PIC.Mom.u_id = sum(PIC.IC.f.*v.*dv,1);
    PIC.Mom.T_id = sum(PIC.IC.f.*v.^2.*dv,1);

    [x, v] = sample_particles_from_f(PIC.IC.f, x(1,:), v(:,1), PIC.config.N);

    class = x*0;

end

%% Prepare Operators

[Gx,Lx] = PIC_operators(PIC.geo);

%% Initial Condition

f = PIC_evaluate_distribution(x,v,PIC.geo);
[a, E] = PIC_getAcc(x, PIC, Gx, Lx);
% 
% figure(1)
% clf
% subplot(3,1,1)
% plot(x(class==0),v(class==0),'.b')
% hold on
% plot(x(class==1),v(class==1),'.r')
% ylim(PIC.geo.range_vn)
% xlim(PIC.geo.range_xn)
% 
% subplot(3,1,2)
% contourf(PIC.geo.xgn,PIC.geo.vgn,f,'LineStyle','none')
% ylim(PIC.geo.range_vn)
% xlim(PIC.geo.range_xn)
% 
% subplot(3,1,3)
% plot(PIC.geo.xn,E)
% xlim(PIC.geo.range_xn)

k = 1;
Results(k).E = E;
Results(k).f = f;
Results(k).t = 0;
Results(k).n = sum(f.*dv,1);
Results(k).u = sum(f.*PIC.geo.vgn.*PIC.geo.dv,1);
Results(k).K = sum(f.*PIC.geo.vgn.^2.*PIC.geo.dv,1);


%% Simulation

t = PIC.time.t_start;

while t < PIC.time.t_end

    disp(t)

    dt_now = 0;

    for i = 1 : PIC.time.dt_results/PIC.time.dt_sim

        dt_now = dt_now + PIC.time.dt_sim;

        t = t + PIC.time.dt_sim;

        v = v + a*PIC.time.dt_sim/2;

        x = x + v*PIC.time.dt_sim;

        x = mod(x, PIC.geo.Ln);

        [a,E] = PIC_getAcc(x, PIC, Gx, Lx);

        v = v + a*PIC.time.dt_sim/2;

    end

    f = PIC_evaluate_distribution(x,v,PIC.geo);
    % 
    % figure(1)
    % clf
    % subplot(3,1,1)
    % plot(x(class==0),v(class==0),'.b')
    % hold on
    % plot(x(class==1),v(class==1),'.r')
    % ylim(PIC.geo.range_vn)
    % xlim(PIC.geo.range_xn)
    % title("time = " + t)
    % 
    % subplot(3,1,2)
    % contourf(PIC.geo.xgn,PIC.geo.vgn,f,'LineStyle','none')
    % ylim(PIC.geo.range_vn)
    % xlim(PIC.geo.range_xn)
    % 
    % subplot(3,1,3)
    % plot(PIC.geo.xn,E)
    % xlim(PIC.geo.range_xn)
    % 
    % drawnow

    k = k + 1;
    Results(k).E = E;
    Results(k).f = f;
    Results(k).t = t;
    Results(k).n = sum(f.*dv,1);
    Results(k).u = sum(f.*PIC.geo.vgn.*PIC.geo.dv,1);
    Results(k).K = sum(f.*PIC.geo.vgn.^2.*PIC.geo.dv,1);


end













