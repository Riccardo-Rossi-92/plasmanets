%% PINN for Vlasov-Poisson: Landau Damping (Forward Problem)
%
% This script trains a CDF-based Physics-Informed Neural Network (PINN)
% to solve the forward Vlasov-Poisson problem for Landau damping using
% the PERTURBATIVE FORMULATION: instead of predicting F directly, the
% network predicts F1(x,v,t) where f = f0 + f1 and f0 is the known
% Maxwellian equilibrium. This allows the network to focus on the
% perturbation, which is orders of magnitude smaller than f0.
%
% Key difference from the two-stream case:
%   - Perturbative output layer: tanh instead of softplus (F1 can be negative)
%
% Reference:
%   R. Rossi, P. Gaudio, M. Gelfusa, A. Murari,
%   "Phase-Space Reconstruction from Macroscopic Observables via
%   Integration-Free Physics-Informed Neural Networks for the
%   Vlasov-Poisson System", Nature Computational Science, [year].
%
% Authors: Riccardo Rossi, Pasquale Gaudio, Michela Gelfusa, Andrea Murari
% Contact: r.rossi@ing.uniroma2.it
% Repository: [GitHub link]
%
% Requirements: MATLAB R[version], Deep Learning Toolbox [version]

%% --- Environment Setup --------------------------------------------------

% Set Config.GPU = 1 to use GPU acceleration (requires Parallel Computing
% Toolbox and a CUDA-compatible GPU). Set to 0 for CPU execution.
Config.GPU = 0;

if canUseGPU && Config.GPU == 1
    gpu = gpuDevice();
    reset(gpu)
    wait(gpu)
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

%% --- Paths --------------------------------------------------------------

addpath src\

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
Norm.t = Norm.L ./ Norm.v;         % = omega_p^-1
Norm.m = species.e.m;
Norm.q = species.e.q;
Norm.E = Norm.L ./ species.e.T;
Norm.f = species.e.n ./ Norm.v;

%% --- Simulation Parameters -----------------------------------------------

% BUG FIX: original code used 'k' for both wavenumber and loop index.
% Renamed wavenumber to k_wave to avoid silent variable overwrite.
k_wave = 0.5;                        % dimensionless wavenumber k*lambda_D

Model = [];

% Spatial domain: L = 2*pi/k (one full wavelength)
% Velocity domain: [-6 v_th, +6 v_th] (wider than two-stream to capture
% the fine phase-mixing structures at late times)
Sim.x_range = [0, 2*pi*Norm.L / k_wave];
Sim.v_range = [-6.*Norm.v, 6.*Norm.v];
Sim.t_range = [0, 10] .* Norm.t;    % T = 10 t_0 (Landau: shorter than two-stream)
Sim.dt      = 1 * Norm.t;           % window width = t_0
Sim.N       = 10;                    % number of sequential time windows

%% --- Main Loop: Sequential Time Windows ----------------------------------

for k = 1 : Sim.N

    fprintf('\n=== Time window %d / %d ===\n', k, Sim.N);

    %% Geometry for current window
    geo.x_range = Sim.x_range;
    geo.v_range = Sim.v_range;
    geo.t_range = [Sim.t_range(1) + (k-1)*Sim.dt, ...
        Sim.t_range(1) +  k   *Sim.dt];

    geo.Nx = 30;
    geo.Nv = 101;
    geo.Nt = 20;

    geo.x = linspace(geo.x_range(1), geo.x_range(2), geo.Nx);
    geo.v = linspace(geo.v_range(1), geo.v_range(2), geo.Nv);
    geo.t = linspace(geo.t_range(1), geo.t_range(2), geo.Nt);

    [geo.xg, geo.vg] = meshgrid(geo.x, geo.v);

    %% Dimensionless coordinates
    geo.xgn      = geo.xg ./ Norm.L;
    geo.vgn      = geo.vg ./ Norm.v;
    geo.tn       = geo.t  ./ Norm.t;
    geo.xn_range = geo.x_range ./ Norm.L;
    geo.vn_range = geo.v_range ./ Norm.v;
    geo.tn_range = geo.t_range ./ Norm.t;

    %% Load PIC Reference Solution
    % PIC data used for plotting and error evaluation only —
    % NOT used as training data for the PINN.
    PIC = load("PIC\pic_landau.mat");

    % Equilibrium Maxwellian f0(v) — known analytic input
    % In dimensionless units: f0 = (1/sqrt(2*pi)) * exp(-v^2/2)
    vth  = 1;   % dimensionless thermal velocity
    f0   = (1 / (sqrt(2*pi) * vth)) * exp(-PIC.Geo.vgn.^2 / (2*vth^2));

    % Plotting setup
    Plot.rate = 50;
    Plot.xn   = PIC.Geo.xgn;
    Plot.vn   = PIC.Geo.vgn;
    Plot.dlX  = dlarray([Plot.xn(:)'; Plot.vn(:)'; Plot.vn(:)'*0], 'CB');

    % Find PIC snapshots closest to window boundaries and midpoint
    t_plot = [geo.tn_range(1), mean(geo.tn_range(1:2)), geo.tn_range(2)];
    t_pic  = linspace(0, 10, 101);
    [~, ind_pic] = min(abs(t_pic' - t_plot));
    Plot.t = t_plot;

    % Colour scale for perturbation plots
    c_min   = min(PIC.Results{ind_pic(1)}.f' - f0, [], 'all');
    c_max   = max(PIC.Results{ind_pic(1)}.f' - f0, [], 'all');

    %% Plot PIC Reference (perturbation f1 = f - f0)
    figure(1); clf;
    rows   = [2, 6, 10];
    titles = ["t = " + PIC.Results{ind_pic(1)}.t, ...
        "t = " + PIC.Results{ind_pic(2)}.t, ...
        "t = " + PIC.Results{ind_pic(3)}.t];
    for p = 1:3
        subplot(3, 4, rows(p));
        contourf(PIC.Geo.xgn, PIC.Geo.vgn, ...
            PIC.Results{ind_pic(p)}.f' - f0, 30, 'LineStyle', 'none');
        grid on; grid minor;
        xlabel('x [λ_D]'); ylabel('v [v_{th}]');
        title("PIC f_1 at " + titles(p) + " [t_0]");
        colorbar(); colormap("jet"); ylim([-3 3]);
    end
    drawnow;

    %% PINN Configuration --------------------------------------------------

    PINN.iteration_per_epochs = 100;
    PINN.physics_mini_batch   = 2000;
    PINN.boundary_mini_batch  = 1000;

    PINN.max_epochs = 2000;
    PINN.min_epochs = 500;

    % Loss weights: [IC, Vlasov, Poisson, Non-negativity, Conservation]
    PINN.weights_label    = ["IC"; "Vlasov"; "Poisson"; "Non-neg"; "Cons"];
    PINN.weights          = [1; 1; 1; 1];

    % Learning rate schedule: warmup + cosine annealing
    % LR = eta0 * epoch/T_warmup        (warmup phase)
    % LR = eta_min + 0.5*(eta0-eta_min)*(1+cos(pi*(epoch-T_w)/(T-T_w))) (cosine)
    PINN.eta0 = 2e-3;
    PINN.eta_min = 5e-5;
    PINN.T_warmup = 25;

    PINN.solver        = "adam";

    %% Prepare Collocation Points ------------------------------------------
    Config.PIC_init = 0;
    [mbq, wx, PINN] = prepare_domain_landau_dec(PINN, geo, Config, Model);

    %% Input Scaling -------------------------------------------------------
    PINN.scaling.x_mi  = (geo.xn_range(2) + geo.xn_range(1)) / 2;
    PINN.scaling.x_std = (geo.xn_range(2) - geo.xn_range(1)) / 2;
    PINN.scaling.v_mi  = (geo.vn_range(2) + geo.vn_range(1)) / 2;
    PINN.scaling.v_std = (geo.vn_range(2) - geo.vn_range(1)) / 2;
    PINN.scaling.t_mi  = (geo.tn_range(2) + geo.tn_range(1)) / 2;
    PINN.scaling.t_std = (geo.tn_range(2) - geo.tn_range(1)) / 2;

    % Loss weighting coefficients (C_j = 1/sigma_j^2)
    % For k=1: compute from initial condition data
    % For k>1: inherit from previous window (perturbation amplitude
    %          remains similar across windows in the linear phase)
    if k == 1
        Xi = next(mbq.initial); reset(mbq.initial);
        PINN.scaling.f0 = double(extractdata(gather(std(Xi(4,:)))));
        PINN.scaling.E0 = double(extractdata(gather(std(Xi(5,:)))));
    else
        % Warm start: reuse scaling from previous window
        PINN.scaling.f0 = new_scaling_f0;
        PINN.scaling.E0 = new_scaling_E0;
    end

    PINN.layer = [20, 20, 20, 20, 20, 20, 20, 20];  % 8 layers × 20 neurons

    Scaling.x  = dlarray([PINN.scaling.x_mi; PINN.scaling.x_std]);
    Scaling.v  = dlarray([PINN.scaling.v_mi; PINN.scaling.v_std]);
    Scaling.t  = dlarray([PINN.scaling.t_mi; PINN.scaling.t_std]);
    Scaling.f0 = PINN.scaling.f0;
    Scaling.E0 = PINN.scaling.E0;

    %% Network Initialisation ----------------------------------------------
    % Warm start: for k>1, use weights from previous window as initial
    % parameters. This exploits temporal continuity of the solution and
    % significantly reduces convergence time for subsequent windows.
    X = next(mbq.physics);
    if k == 1
        [~, ~, parameters] = NN_VlasovPoisson_dec(...
            X(1,:), X(2,:), X(3,:), 0, [], PINN.layer, Scaling);
    else
        % Warm start from previous window
        parameters = Model{k-1}.parameters;
        fprintf('  Warm start from window %d\n', k-1);
    end

    % Transfer parameters to GPU if available
    if Config.GPU == 1
        parameters = dlupdate(@gpuArray, parameters);
    end

    % Verify forward pass
    [~, ~] = NN_VlasovPoisson_dec(...
        X(1,:), X(2,:), X(3,:), 1, parameters, PINN.layer, Scaling);
    clear X; reset(mbq.physics);

    %% Accelerated Model Gradient ------------------------------------------
    modelgradient = dlaccelerate(@modelgradient_landau_dec);

    %% Output File Setup ---------------------------------------------------
    basename = "results\Landau_tdd_t" + geo.tn_range(1);
    counter  = 1;
    filename = sprintf('%s%d.mat', basename, counter);
    while exist(filename, 'file')
        counter  = counter + 1;
        filename = sprintf('%s%d.mat', basename, counter);
    end
    clear counter basename;

    %% Initialise Training State -------------------------------------------

    iteration      = 0;
    averageSqGrad  = [];
    averageGrad    = [];

    % % Learning rate schedule parameters
    T_warmup = PINN.T_warmup;                    % warmup epochs
    maxEpochs = PINN.max_epochs;
    eta0      = PINN.eta0;                   % peak learning rate
    eta_min   = PINN.eta_min;                   % minimum learning rate
    T         = maxEpochs;

    % Error tracking matrix [3 time instants x maxEpochs]
    RMSE = zeros(3, maxEpochs);
    RMSE_E = zeros(3, maxEpochs);

    weights = PINN.weights;

    %% Training Loop -------------------------------------------------------

    epoch       = 0;
    convergence = false;
    LearningRate = eta_min;             % will be updated by schedule

    while ~convergence

        tic;
        epoch = epoch + 1;

        %% Inner loop: gradient steps per epoch
        for i = 1 : PINN.iteration_per_epochs

            iteration = iteration + 1;

            Xp = next(mbq.physics);
            Xg = next(mbq.gridX);
            Xi = next(mbq.initial);

            % Note: Xp columns 6 and 7 carry f0-related quantities
            % needed by the perturbative Vlasov loss
            [gradients, Loss, Losses] = dlfeval(modelgradient, ...
                parameters, PINN, Scaling, ...
                Xp(1,:), Xp(2,:), Xp(3,:), Xp(7,:), ...
                Xi(1,:), Xi(2,:), Xi(3,:), Xi(4,:), Xi(5,:), ...
                Xg(1,:), Xg(2,:), Xg(3,:), wx, ...
                weights);

            if isnan(double(extractdata(gather(Loss))))
                warning('NaN loss at epoch %d, iteration %d. Aborting.', epoch, iteration);
                return;
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
        reset(mbq.initial);

        %% Periodic Evaluation and Plotting --------------------------------
        if mod(epoch, Plot.rate) == 0

            % Loss curves
            subplot(3,4,1); hold on;
            plot(epoch, Losses(1).*weights(1), '.r', 'MarkerSize', 16);
            plot(epoch, Losses(2).*weights(2), '.b', 'MarkerSize', 16);
            plot(epoch, Losses(3).*weights(3), '.g', 'MarkerSize', 16);
            plot(epoch, Losses(4).*weights(4), '.m', 'MarkerSize', 16);
            set(gca, 'YScale', 'log'); grid on; grid minor;
            title('Weighted losses'); legend('IC','Vlasov','Poisson','Non-neg');

            subplot(3,4,5); hold on;
            plot(epoch, Loss, '.b', 'MarkerSize', 16);
            set(gca, 'YScale', 'log'); grid on; grid minor;
            title('Total loss');

            % subplot(3,4,9); hold on;
            % plot(epoch, Losses(5), '.r', 'MarkerSize', 16);
            % set(gca, 'YScale', 'log'); grid on; grid minor;
            % title('Conservation loss');

            % Evaluate PINN predictions at three time instants
            for p = 1:3
                Plot.dlX(3,:) = PIC.Results{ind_pic(p)}.t;
                [f1, E1, ~] = dlfeval(@NN_VlasovPoisson_Predict, ...
                    Plot.dlX(1,:), Plot.dlX(2,:), Plot.dlX(3,:), ...
                    parameters, PINN.layer, Scaling);

                f1 = reshape(f1, size(Plot.xn));
                E1 = reshape(E1, size(Plot.xn));

                % Reference: PIC perturbation f1 = f - f0
                fn = PIC.Results{ind_pic(p)}.f' - f0;

                % Relative L2 error vs PIC reference
                L2f(p, epoch) = norm(f1(:) - fn(:)) / norm(fn(:));
                L2E(p,epoch) = norm(E1(1,:)'-PIC.Results{ind_pic(p)}.E(:)) /...
                    norm(PIC.Results{ind_pic(p)}.E(:));

                % Correlation coefficient — more meaningful than L2 for
                % small perturbations where noise can dominate the norm
                R(p) = corr(f1(:), fn(:));

                subplot(3, 4, rows(p) + 1);
                contourf(Plot.xn, Plot.vn, f1, 30, 'LineStyle', 'none');
                colorbar(); colormap("jet"); ylim([-3, 3]);
                grid on; grid minor;
                xlabel('x [λ_D]'); ylabel('v [v_{th}]');
                title(sprintf('PINN f_1 at %s [t_0], r=%.3f', titles(p), R(p)));

                subplot(3, 4, rows(p) + 2);
                hold off;
                plot(PIC.Geo.xn, PIC.Results{ind_pic(p)}.E, 'b', ...
                    'DisplayName', 'PIC');
                hold on;
                plot(Plot.xn(1,:), E1(1,:), 'r--', 'DisplayName', 'PINN');
                legend; grid on;
                title("E(x) at " + titles(p));
            end

            drawnow;
            fprintf('  L2 f: [%.4f, %.4f, %.4f]\n',...
                L2f(1,epoch), L2f(2,epoch), L2f(3,epoch));

            % Plot total loss
            subplot(3,4,9)
            plot(L2E(:,1:epoch)', '.', 'MarkerSize', 16);
            set(gca, 'YScale', 'log'); grid on; grid minor;
            title('Total loss');
            drawnow;
        end

        %% Learning Rate Schedule ------------------------------------------
        % Phase 1 (epoch < T_warmup): linear warmup from 0 to eta0
        % Phase 2 (epoch >= T_warmup): cosine annealing from eta0 to eta_min
        if epoch < T_warmup
            LearningRate = eta0 * epoch / T_warmup;
        else
            LearningRate = eta_min + 0.5 * (eta0 - eta_min) * ...
                (1 + cos(pi * (epoch - T_warmup) / (T - T_warmup)));
        end

        %% Convergence Check -----------------------------------------------
        if epoch >= PINN.max_epochs
            convergence = true;
            fprintf('  Window %d converged after %d epochs.\n', k, epoch);
        end

        %% Progress Report -------------------------------------------------
        Loss_scalar = double(extractdata(gather(sum(weights.*Losses))));
        fprintf('  k=%d | epoch=%d | loss=%.4e | LR=%.2e | time=%.1fs\n', ...
            k, epoch, Loss_scalar, LearningRate, toc);

    end % while ~convergence

    %% Store Window Results ------------------------------------------------
    Model{k}.geo        = geo;
    Model{k}.parameters = parameters;
    Model{k}.PINN       = PINN;
    Model{k}.Scaling    = Scaling;

    %% Update Scaling for Next Window --------------------------------------
    % Use RMS of current predictions as scaling for next window.
    % Factor of 3 on f1 provides margin for perturbation growth in the
    % nonlinear phase — prevents underscaling when vortices develop.
    [f1_out, E1_out, ~] = dlfeval(@NN_VlasovPoisson_Predict, ...
        Xp(1,:), Xp(2,:), Xp(3,:), parameters, PINN.layer, Scaling);

    new_scaling_f0 = 3 * double(extractdata(gather(sqrt(mean(f1_out.^2)))));
    new_scaling_E0 =     double(extractdata(gather(sqrt(mean(E1_out.^2)))));

    fprintf('  Updated scaling: f0=%.4f (x3), E0=%.4f\n', ...
        new_scaling_f0/3, new_scaling_E0);

    %% Save Results --------------------------------------------------------
    save(filename, 'Model', 'PINN', 'geo', 'Scaling');
    fprintf('  Results saved to: %s\n', filename);

end % for k = 1:Sim.N

fprintf('\n=== Training complete for all %d time windows ===\n', Sim.N);