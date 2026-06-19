%% PINN for Vlasov-Poisson: Inverse Problem with Enriched Descriptors
%
% This script trains a CDF-based PINN to solve the INVERSE problem
% using an ENRICHED set of macroscopic observables: electric field E,
% electron density n, mean velocity u, and temperature T.
%
% Compared to the basic inverse problem (main_inverse.m), this version
% uses two additional cumulative variables G and H:
%   G(x,v,t) = integral_{-inf}^{v} f(x,v',t) * v'  dv'  → mean velocity
%   H(x,v,t) = integral_{-inf}^{v} f(x,v',t) * v'^2 dv' → temperature
%
% All four moments are computed WITHOUT numerical integration — they are
% recovered as boundary evaluations of G and H at v_max:
%   n(x,t) = F(x, v_max, t)
%   u(x,t) = G(x, v_max, t) / n(x,t)
%   T(x,t) = H(x, v_max, t) / n(x,t) - u(x,t)^2
%
% This property is essential for future extensions to higher-dimensional
% systems and to the Vlasov-Maxwell framework.
%
% Key difference from main_inverse.m:
%   - Network: NN_VlasovPoisson_dec_m3_v_T (3 output CDFs: F, G, H)
%   - Loss: 6 terms (adds L_u and L_T to the 5 of basic inverse)
%   - Measurement data: Xm has 7 columns (adds u and T observations)
%   - Plotting: dual y-axis showing E and u simultaneously
%
% Reference:
%   R. Rossi, P. Gaudio, M. Gelfusa, A. Murari,
%   "Phase-Space Reconstruction from Macroscopic Observables via
%   Integration-Free Physics-Informed Neural Networks for the
%   Vlasov-Poisson System", Nature Computational Science, [year].
%
% Authors: Riccardo Rossi, Pasquale Gaudio, Michela Gelfusa, Andrea Murari
% Contact: r.rossi@ing.uniroma2.it

%% --- Environment Setup ---------------------------------------------------

Config.GPU = 0;

if canUseGPU && Config.GPU == 1
    gpu = gpuDevice(); reset(gpu); wait(gpu);
    clear all; clc;
    Config.GPU = 1;
    Config.environment = "gpu";
    disp("GPU selected as working environment")
else
    clear all; clc;
    Config.GPU = 0;
    Config.environment = "cpu";
    disp("CPU selected as working environment")
end

warning off

%% --- Paths ---------------------------------------------------------------

addpath src\
addpath src\utilities\
addpath src\solutions\
addpath src\architecture\
addpath src\modelgradient\

%% --- Physical Constants --------------------------------------------------

c = constants();

%% --- Species Definition --------------------------------------------------

species.e.n       = 1e19;
species.e.T       = 100;
species.e.omega   = sqrt(species.e.n .* c.qe.^2 ./ (c.me .* c.eps0));
species.e.lambdaD = sqrt(abs(c.eps0 .* species.e.T ./ (c.qe .* species.e.n)));
species.e.m       = c.me;
species.e.q       = c.qe;
species.e.vth     = sqrt(2 .* abs(c.qe) .* species.e.T ./ c.me);

%% --- Normalisation Factors -----------------------------------------------

Norm.L = species.e.lambdaD;
Norm.v = species.e.vth;
Norm.t = Norm.L ./ Norm.v;
Norm.m = species.e.m;
Norm.q = species.e.q;
Norm.E = Norm.L ./ species.e.T;
Norm.f = species.e.n ./ Norm.v;

%% --- Inverse Problem Configuration --------------------------------------
%
% KEY PARAMETER: observation window DT = t_range(2) - t_range(1)
% Controls the observability of the inverse problem:
%   DT = 0.1 t0  → reconstruction fails (underdetermined)
%   DT = 1   t0  → partial reconstruction
%   DT = 5   t0  → good reconstruction
%   DT = 19  t0  → excellent reconstruction (well-posed)
%
% To reproduce Fig. 4 of the paper, run this script for each DT value:
%   DT_values = [0.1, 1, 5, 10, 19];

DT = 1;           % observation window width [t0] — change to reproduce Fig. 4
t0_target = 11;   % [t0] time at which f is reconstructed

Sim.x_range = [0, 2*pi*Norm.L];
Sim.v_range = [-5.*Norm.v, 5.*Norm.v];
Sim.t_range = [t0_target - DT, t0_target] .* Norm.t;

%% --- Geometry (single window — no time decomposition) -------------------

geo.x_range = Sim.x_range;
geo.v_range = Sim.v_range;
geo.t_range = [Sim.t_range(1), Sim.t_range(end)];

geo.Nx = 30;
geo.Nv = 101;
geo.Nt = 20;

geo.x = linspace(geo.x_range(1), geo.x_range(2), geo.Nx);
geo.v = linspace(geo.v_range(1), geo.v_range(2), geo.Nv);
geo.t = linspace(geo.t_range(1), geo.t_range(2), geo.Nt);

[geo.xg, geo.vg] = meshgrid(geo.x, geo.v);

geo.xgn      = geo.xg ./ Norm.L;
geo.vgn      = geo.vg ./ Norm.v;
geo.tn       = geo.t  ./ Norm.t;
geo.xn_range = geo.x_range ./ Norm.L;
geo.vn_range = geo.v_range ./ Norm.v;
geo.tn_range = geo.t_range ./ Norm.t;

fprintf('Inverse problem (enriched): DT = %.2f t0 | window [%.2f, %.2f] t0\n', ...
        DT, geo.tn_range(1), geo.tn_range(2));

%% --- Load PIC Reference (for plotting and error evaluation only) ---------
% NOTE: PIC data are used ONLY for visual comparison and error metrics.
% The PINN never sees f from PIC during training — only E, n, u, T
% measurements derived from the PIC solution (with optional noise).
%
% run06_orders.mat contains higher-order moments (u, T) in addition
% to the standard fields available in run06.mat.

PIC = load("PIC\pic_twostream.mat");

f0   = PIC.PIC.IC.f0;   % equilibrium distribution (for reference)
vth  = 0.2;
vb   = 1;

% Find PIC snapshots within the observation window
t_plot = [geo.tn_range(1), mean(geo.tn_range(1:2)), geo.tn_range(2)];
t_pic  = linspace(PIC.PIC.time.t_start, PIC.PIC.time.t_end, length(PIC.Results));
[~, ind_pic] = min(abs(t_pic' - t_plot));
Plot.t = t_plot;

% Plotting setup
Plot.rate = 5;
Plot.xn   = PIC.PIC.geo.xgn;
Plot.vn   = PIC.PIC.geo.vgn;
Plot.dlX  = dlarray([Plot.xn(:)'; Plot.vn(:)'; Plot.vn(:)'*0], 'CB');

%% Initial reference plot (PIC target distribution)
figure(1); clf;
rows   = [2, 6, 10];
titles = ["t = " + t_plot(1), "t = " + mean(t_plot(1:2)), "t = " + t_plot(3)];
for p = 1:3
    subplot(3, 4, rows(p));
    contourf(PIC.PIC.geo.xgn, PIC.PIC.geo.vgn, ...
             PIC.Results(ind_pic(p)).f, 30, 'LineStyle', 'none');
    grid on; grid minor;
    xlabel('x [λ_D]'); ylabel('v [v_{th}]');
    title("PIC " + titles(p) + " [t_0]");
    colorbar(); colormap("jet"); ylim([-3, 3]);
end
drawnow;

%% --- PINN Configuration --------------------------------------------------

PINN.iteration_per_epochs    = 100;
PINN.physics_mini_batch      = 2000;
PINN.boundary_mini_batch     = 1000;
PINN.measurements_mini_batch = 1000;

PINN.max_epochs = 200000;
PINN.min_epochs = 500;

% Loss weights: [Measurements, Vlasov, Poisson, Non-neg, Conservation, u+T]
% 6 losses for enriched version (vs 5 for basic inverse)
PINN.weights_label = ["Measurements"; "Vlasov"; "Poisson"; "Cons"; "Non-neg"];
PINN.weights       = [1; 1; 1; 1; 1; 1e-2];

PINN.learning_rate     = 5e-3;
PINN.learning_rate_min = 1e-6;
PINN.learning_rate_max = 5e-3;
PINN.solver            = "adam";
PINN.patience          = 200;

%% --- Prepare Collocation and Measurement Points -------------------------
% prepare_domain_twostream_dec_inverse_orders builds:
%   mbq.physics      — collocation points for Vlasov and Poisson losses
%   mbq.gridX        — spatial grid for conservation loss
%   mbq.measurements — measurement points with 7 columns:
%                      [x, t, E_obs, n_obs, u_obs, T_obs, noise_flag]

Config.PIC_init = 0;
Model = [];
[mbq, wx, PINN] = prepare_domain_twostream_inverse_u_T(PINN, geo, Config, Model);

%% --- Input Scaling -------------------------------------------------------

PINN.scaling.x_mi  = (geo.xn_range(2) + geo.xn_range(1)) / 2;
PINN.scaling.x_std = (geo.xn_range(2) - geo.xn_range(1)) / 2;
PINN.scaling.v_mi  = (geo.vn_range(2) + geo.vn_range(1)) / 2;
PINN.scaling.v_std = (geo.vn_range(2) - geo.vn_range(1)) / 2;
PINN.scaling.t_mi  = (geo.tn_range(2) + geo.tn_range(1)) / 2;
PINN.scaling.t_std = (geo.tn_range(2) - geo.tn_range(1)) / 2;

Xi = next(mbq.initial); reset(mbq.initial);
PINN.scaling.f0 = 1;
PINN.scaling.E0 = double(extractdata(gather(std(Xi(5,:)))));

PINN.layer = [20, 20, 20, 20, 20, 20, 20, 20];

Scaling.x  = dlarray([PINN.scaling.x_mi; PINN.scaling.x_std]);
Scaling.v  = dlarray([PINN.scaling.v_mi; PINN.scaling.v_std]);
Scaling.t  = dlarray([PINN.scaling.t_mi; PINN.scaling.t_std]);
Scaling.f0 = PINN.scaling.f0;
Scaling.E0 = PINN.scaling.E0;

%% --- Network Initialisation ----------------------------------------------
% NN_VlasovPoisson_dec_m3_v_T extends the base network with two additional
% output CDFs:
%   F(x,v,t) = integral f dv'         → density (as in basic version)
%   G(x,v,t) = integral f*v' dv'      → numerator of mean velocity
%   H(x,v,t) = integral f*v'^2 dv'    → numerator of temperature
%
% Preconditioned weights loaded if available to accelerate convergence.

X = next(mbq.physics);
[~, ~, parameters] = NN_VlasovPoisson_dec_u_T(...
    X(1,:), X(2,:), X(3,:), 0, [], PINN.layer, Scaling);

% Load preconditioned weights if available
if exist("preconditioned_v_T.mat", "file")
    load("preconditioned_v_T.mat");
    disp("Loaded preconditioned weights (v_T version).");
end

% Transfer to GPU if available
if Config.GPU == 1
    parameters = dlupdate(@gpuArray, parameters);
end

% Verify forward pass
[~, ~] = NN_VlasovPoisson_dec_u_T(...
    X(1,:), X(2,:), X(3,:), 1, parameters, PINN.layer, Scaling);
clear X; reset(mbq.physics);

%% --- Accelerated Model Gradient ------------------------------------------
modelgradient = dlaccelerate(@modelgradient_inverse_u_T);

%% --- Training State Initialisation --------------------------------------

iteration     = 0;
averageSqGrad = [];
averageGrad   = [];

% BUG FIX: T_warmup and maxEpochs now correctly tied to PINN.max_epochs
% (original code had T_warmup=10, maxEpochs=200 with PINN.max_epochs=200000,
% causing cosine annealing to flatten after only 200 epochs)
T_warmup  = round(0.05 * PINN.max_epochs);   % 5% warmup = 10000 epochs
maxEpochs = PINN.max_epochs;                  % = 200000
eta0      = PINN.learning_rate;               % = 5e-3
eta_min   = PINN.learning_rate_min;           % = 1e-6
T         = maxEpochs;

LearningRate = eta0 / T_warmup;   % start low and warm up

RMSE     = zeros(3, PINN.max_epochs);
M_values = zeros(length(PINN.weights), PINN.max_epochs);

%% --- Output File Setup ---------------------------------------------------
basename = sprintf('results\\Inverse_enriched_DT%.2f_t%.1f', DT, t0_target);
counter  = 1;
filename = sprintf('%s_%d.mat', basename, counter);
while exist(filename, 'file')
    counter  = counter + 1;
    filename = sprintf('%s_%d.mat', basename, counter);
end
clear counter basename;

%% --- Loss Weights Initialisation -----------------------------------------
weights = PINN.weights;

%% --- Training Loop -------------------------------------------------------

epoch       = 0;
convergence = false;

while ~convergence

    tic;
    epoch = epoch + 1;

    %% Inner loop: gradient steps per epoch
    for i = 1 : PINN.iteration_per_epochs

        iteration = iteration + 1;

        Xp = next(mbq.physics);
        Xg = next(mbq.gridX);

        % Shuffle measurements when exhausted (avoid temporal bias)
        if ~hasdata(mbq.measurements)
            shuffle(mbq.measurements);
        end
        Xm = next(mbq.measurements);

        % Xm columns: [x, t, E_obs, n_obs, u_obs, T_obs, noise_flag]
        % 7 columns vs 5 in basic inverse — adds u and T measurements
        [gradients, Loss, Losses] = dlfeval(modelgradient, ...
            parameters, PINN, Scaling, ...
            Xp(1,:), Xp(2,:), Xp(3,:), ...
            Xm(1,:), Xm(2,:), Xm(3,:), Xm(4,:), Xm(5,:), Xm(6,:), Xm(7,:), ...
            Xg(1,:), Xg(2,:), Xg(3,:), wx, ...
            weights);

        % Check for numerical instability (every 10 iter to reduce GPU transfer)
        if mod(iteration, 10) == 0
            if isnan(double(extractdata(gather(Loss))))
                warning('NaN loss at epoch %d. Aborting.', epoch);
                return;
            end
        end

        if PINN.solver == "adam"
            [parameters, averageGrad, averageSqGrad] = adamupdate(...
                parameters, gradients, averageGrad, averageSqGrad, ...
                iteration, LearningRate);
        elseif PINN.solver == "rms"
            [parameters, averageSqGrad] = rmspropupdate(...
                parameters, gradients, averageSqGrad, LearningRate);
        end

    end

    reset(mbq.physics);
    reset(mbq.gridX);

    %% Periodic Evaluation and Plotting ------------------------------------
    if mod(epoch, Plot.rate) == 0

        % Loss curves (6 losses for enriched version)
        subplot(3,4,1); hold on;
        plot(epoch, Losses(1).*weights(1), '.r', 'MarkerSize', 16);
        plot(epoch, Losses(2).*weights(2), '.b', 'MarkerSize', 16);
        plot(epoch, Losses(3).*weights(3), '.g', 'MarkerSize', 16);
        plot(epoch, Losses(4).*weights(4), '.m', 'MarkerSize', 16);
        plot(epoch, Losses(5).*weights(5), 'sr', 'MarkerSize', 16);
        set(gca, 'YScale', 'log'); grid on; grid minor;
        title('Losses'); legend('Meas','Vlasov','Poisson','Non-neg','Cons');

        subplot(3,4,5); hold on;
        plot(epoch, Loss, '.b', 'MarkerSize', 16);
        set(gca, 'YScale', 'log'); grid on; grid minor;
        title('Total loss');

        % Evaluate reconstructed f, E, u at three time instants
        % NN_VlasovPoisson_Predict_v_T returns: f, E, F (density CDF),
        % Fv (velocity CDF = G at v_max), FT (temperature CDF = H at v_max)
        for p = 1:3
            Plot.dlX(3,:) = PIC.Results(ind_pic(p)).t;
            [f_pred, E_pred, ~, Fv_pred, ~] = dlfeval(@NN_VlasovPoisson_Predict_u_T, ...
                Plot.dlX(1,:), Plot.dlX(2,:), Plot.dlX(3,:), ...
                parameters, PINN.layer, Scaling);

            f_pred  = reshape(f_pred,  size(Plot.xn));
            E_pred  = reshape(E_pred,  size(Plot.xn));
            Fv_pred = reshape(Fv_pred, size(Plot.xn));
            fn      = PIC.Results(ind_pic(p)).f;

            % Relative L2 error vs PIC target
            RMSE(p, epoch) = norm(f_pred(:) - fn(:)) / norm(fn(:));

            % Phase-space plot
            subplot(3, 4, rows(p) + 1);
            contourf(Plot.xn, Plot.vn, f_pred, 30, 'LineStyle', 'none');
            colorbar(); colormap("jet"); ylim([-3, 3]);
            grid on; grid minor;
            xlabel('x [λ_D]'); ylabel('v [v_{th}]');
            title(sprintf('PINN %s | L2=%.3f', titles(p), RMSE(p,epoch)));

            % Dual y-axis: E (left, blue) and u (right, red)
            % This allows simultaneous assessment of both observable sets
            subplot(3, 4, rows(p) + 2);
            yyaxis left;
            hold off;
            plot(PIC.PIC.geo.xn', PIC.Results(ind_pic(p)).E, '-b', ...
                 'DisplayName', 'E PIC');
            hold on;
            plot(Plot.xn(1,:), E_pred(1,:), 'ob', ...
                 'DisplayName', 'E PINN');
            ylabel('E [kT_e/(eλ_D)]');

            yyaxis right;
            hold off;
            plot(PIC.PIC.geo.xn, PIC.Results(ind_pic(p)).u, '-r', ...
                 'DisplayName', 'u PIC');
            hold on;
            plot(Plot.xn(1,:), Fv_pred(end,:), 'or', ...
                 'DisplayName', 'u PINN');
            ylabel('u [v_{th}]');
            grid on; title("E and u at " + titles(p));
        end

        drawnow;
        fprintf('  L2 errors: [%.4f, %.4f, %.4f]\n', ...
                RMSE(1,epoch), RMSE(2,epoch), RMSE(3,epoch));
    end

    %% Cosine Annealing with Warmup ----------------------------------------
    if epoch < T_warmup
        LearningRate = eta0 * epoch / T_warmup;
    else
        LearningRate = eta_min + 0.5 * (eta0 - eta_min) * ...
            (1 + cos(pi * (epoch - T_warmup) / (T - T_warmup)));
    end

    if epoch >= PINN.max_epochs
        convergence = true;
    end

    %% Progress Report -----------------------------------------------------
    fprintf('DT=%.2f | ep=%d/%d | loss=%.3e | LR=%.1e | t=%.1fs\n', ...
            DT, epoch, PINN.max_epochs, ...
            double(extractdata(gather(Loss))), LearningRate, toc);

    M_values(:, epoch) = double(extractdata(gather(Losses)));

end % while ~convergence

fprintf('\n=== Enriched inverse problem complete. DT=%.2f t0 ===\n', DT);
fprintf('Final L2 errors: [%.4f, %.4f, %.4f]\n', ...
        RMSE(1,epoch), RMSE(2,epoch), RMSE(3,epoch));

%% --- Save Results --------------------------------------------------------
save(filename, 'parameters', 'PINN', 'geo', 'Scaling', 'RMSE', 'DT');
fprintf('Results saved to: %s\n', filename);