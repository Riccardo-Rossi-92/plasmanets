%% PINN for Vlasov-Poisson: Two-Stream Instability (Forward Problem)
%
% This script trains a CDF-based Physics-Informed Neural Network (PINN)
% to solve the forward Vlasov-Poisson problem for the two-stream instability.
%
% The simulation domain [0, T] is decomposed into N sequential time windows.
% Each window is solved by a dedicated PINN; the output of the i-th network
% at its final time provides the initial condition for the (i+1)-th network.
%
% Reference:
%   R. Rossi, P. Gaudio, M. Gelfusa, A. Murari,
%   "Phase-Space Reconstruction from Macroscopic Observables via
%   Integration-Free Physics-Informed Neural Networks for the
%   Vlasov-Poisson System", Nature Computational Science, [year].
%
% Contact: r.rossi@ing.uniroma2.it
%
% Hardware: CPU (tested on Intel Core Ultra 7 155H)

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

%% --- Physical Constants -------------------------------------------------

c = constants();

%% --- Species Definition (electrons, stationary ion background) ----------

species.e.n      = 1e19;                                    % [m^-3] density
species.e.T      = 100;                                     % [eV] temperature
species.e.omega  = sqrt(species.e.n .* c.qe.^2 ./ ...
    (c.me .* c.eps0));                  % [rad/s] plasma frequency
species.e.lambdaD = sqrt(abs(c.eps0 .* species.e.T ./ ...
    (c.qe .* species.e.n)));       % [m] Debye length
species.e.m      = c.me;                                    % [kg] electron mass
species.e.q      = c.qe;                                    % [C] electron charge
species.e.vth    = sqrt(2 .* abs(c.qe) .* ...
    species.e.T ./ c.me);              % [m/s] thermal velocity

%% --- Normalisation Factors ----------------------------------------------
% Dimensionless variables: x/lambda_D, v/v_th, t*omega_p
% See Methods section of the paper for full adimensionalisation.

Norm.L = species.e.lambdaD;
Norm.v = species.e.vth;
Norm.t = Norm.L ./ Norm.v;            % = omega_p^-1
Norm.m = species.e.m;
Norm.q = species.e.q;
Norm.E = Norm.L ./ species.e.T;       % electric field normalisation
Norm.f = species.e.n ./ Norm.v;       % distribution function normalisation

%% --- Simulation Parameters ----------------------------------------------

Model = [];                            % will store trained networks for each window

Sim.x_range = [0, 2*pi*Norm.L];        % spatial domain [m]
Sim.v_range = [-5.*Norm.v, 5.*Norm.v]; % velocity domain [m/s]
Sim.t_range = [0, 20] .* Norm.t;       % total time domain: T = 20 t_0
Sim.dt      = Norm.t;                  % window width: t_0 = L/v_th
Sim.N       = 20;                      % number of sequential time windows

%% --- Main Loop: Sequential Time Windows ----------------------------------

for k = 1 : Sim.N   

    fprintf('\n=== Time window %d / %d ===\n', k, Sim.N);

    %% Geometry for current window (dimensional)
    geo.x_range = Sim.x_range;
    geo.v_range = Sim.v_range;
    geo.t_range = [Sim.t_range(1) + (k-1)*Sim.dt, ...
        Sim.t_range(1) +  k   *Sim.dt];

    % Evaluation grid (for plotting and error computation only —
    % collocation points are sampled separately via Sobol sequences)
    geo.Nx = 30;
    geo.Nv = 101;
    geo.Nt = 20;

    geo.x = linspace(geo.x_range(1), geo.x_range(2), geo.Nx);
    geo.v = linspace(geo.v_range(1), geo.v_range(2), geo.Nv);
    geo.t = linspace(geo.t_range(1), geo.t_range(2), geo.Nt);

    [geo.xg, geo.vg] = meshgrid(geo.x, geo.v);

    %% Dimensionless coordinates
    geo.xgn     = geo.xg ./ Norm.L;
    geo.vgn     = geo.vg ./ Norm.v;
    geo.tn      = geo.t  ./ Norm.t;
    geo.xn_range = geo.x_range ./ Norm.L;
    geo.vn_range = geo.v_range ./ Norm.v;
    geo.tn_range = geo.t_range ./ Norm.t;

    %% Load PIC Reference Solution
    % PIC data are used for: (1) initial condition at k=1, (2) plotting,
    % (3) error evaluation. They are NOT used as training data for the PINN.
    PIC = load("PIC\pic_run_1.mat");

    % Plotting grid (matches PIC output grid)
    Plot.rate = 50;                     % plot every N epochs
    Plot.xn   = PIC.PIC.geo.xgn;
    Plot.vn   = PIC.PIC.geo.vgn;
    Plot.dlX  = dlarray([Plot.xn(:)'; Plot.vn(:)'; Plot.vn(:)'*0], 'CB');

    % Initial condition parameters (two-stream: two counter-propagating beams)
    vth = 0.2;
    vb  = 1.0;
    f0  = PIC.PIC.IC.f0;               % equilibrium distribution

    % Find PIC snapshots closest to window boundaries and midpoint
    t_plot = [geo.tn_range(1), mean(geo.tn_range(1:2)), geo.tn_range(2)];
    t_pic  = linspace(PIC.PIC.time.t_start, PIC.PIC.time.t_end, ...
        length(PIC.Results));
    [~, ind_pic] = min(abs(t_pic' - t_plot));
    Plot.t = t_plot;

    % Colour scale for phase-space plots (consistent across panels)
    c_min   = min(PIC.Results(ind_pic(1)).f - f0, [], 'all');
    c_max   = max(PIC.Results(ind_pic(1)).f - f0, [], 'all');

    %% Plot PIC Reference (for visual comparison during training)
    figure(1); clf;
    titles = ["t = " + PIC.Results(ind_pic(1)).t, ...
        "t = " + PIC.Results(ind_pic(2)).t, ...
        "t = " + PIC.Results(ind_pic(3)).t];
    rows = [2, 6, 10];
    for p = 1:3
        subplot(3, 4, rows(p));
        contourf(PIC.PIC.geo.xgn, PIC.PIC.geo.vgn, ...
            PIC.Results(ind_pic(p)).f, 30, 'LineStyle', 'none');
        grid on; grid minor;
        xlabel('x [λ_D]'); ylabel('v [v_{th}]');
        title("PIC " + titles(p) + " [t_0]");
        colorbar(); colormap("jet"); ylim([-3 3]);
    end
    drawnow;

    %% PINN Configuration --------------------------------------------------

    % Mini-batch sizes for each loss term
    PINN.iteration_per_epochs  = 100;   % gradient steps per epoch
    PINN.physics_mini_batch    = 2000;  % collocation points for physics loss
    PINN.boundary_mini_batch   = 1000;  % points for boundary/IC loss

    % Training stopping criteria
    PINN.max_epochs = 3000;
    PINN.min_epochs = 500;

    % Loss weights: [IC, Vlasov, Poisson, Non-negativity, Conservation]
    % which_adaptive: indices of losses with adaptive weighting
    PINN.weights_label    = ["IC"; "Vlasov"; "Poisson"; "Cons";  "Non-neg"];
    PINN.weights          = [1; 1; 1; 1; 1e-2];

    % Learning rate schedule: warmup + cosine annealing
    % LR = eta0 * epoch/T_warmup        (warmup phase)
    % LR = eta_min + 0.5*(eta0-eta_min)*(1+cos(pi*(epoch-T_w)/(T-T_w))) (cosine)
    PINN.eta0 = 5e-3;
    PINN.eta_min = 1e-4;
    PINN.T_warmup = 25;

    % Optimiser settings
    PINN.solver        = "adam";

    %% Prepare Collocation Points ------------------------------------------
    Config.PIC_init = 0;
    [mbq, wx, PINN] = prepare_domain_twostream_dec(PINN, geo, Config, Model);

    %% Input Scaling (for numerical stability) -----------------------------
    % Inputs are standardised to zero mean and unit standard deviation
    % over the training domain before entering the network.

    PINN.scaling.x_mi  = (geo.xn_range(2) + geo.xn_range(1)) / 2;
    PINN.scaling.x_std = (geo.xn_range(2) - geo.xn_range(1)) / 2;
    PINN.scaling.v_mi  = (geo.vn_range(2) + geo.vn_range(1)) / 2;
    PINN.scaling.v_std = (geo.vn_range(2) - geo.vn_range(1)) / 2;
    PINN.scaling.t_mi  = (geo.tn_range(2) + geo.tn_range(1)) / 2;
    PINN.scaling.t_std = (geo.tn_range(2) - geo.tn_range(1)) / 2;

    % Loss weighting coefficients based on initial std of f and E
    % (see Methods section: C_j = 1/sigma_j^2)
    if k == 1
        Xi = next(mbq.initial); reset(mbq.initial);
        PINN.scaling.f0 = 1;
        PINN.scaling.E0 = double(extractdata(gather(std(Xi(5,:)))));
    end
    % For k > 1, scaling is inherited from the previous window (no update
    % needed since the solution amplitude is similar across windows).

    %% Network Architecture ------------------------------------------------
    % V-PINN and P-PINN: fully connected, 8 layers x 20 neurons, tanh activation.
    % Input: (cos(2πx/L), sin(2πx/L), v, t) for V-PINN; (cos, sin, t) for P-PINN.
    % Output layer: softplus * (v - v_min) to enforce F >= 0 and F(v_min) = 0.

    PINN.layer = [20, 20, 20, 20, 20, 20, 20, 20];  % 8 hidden layers, 20 neurons
    % PINN.layer = [40,40,40,40];  % 4 hidden layers, 40 neurons
    % PINN.layer = [80,80,80,80];  % 4 hidden layers, 80 neurons

    Scaling.x  = dlarray([PINN.scaling.x_mi; PINN.scaling.x_std]);
    Scaling.v  = dlarray([PINN.scaling.v_mi; PINN.scaling.v_std]);
    Scaling.t  = dlarray([PINN.scaling.t_mi; PINN.scaling.t_std]);
    Scaling.f0 = PINN.scaling.f0;
    Scaling.E0 = PINN.scaling.E0;

    % Initialise or reinitialise network parameters
    X = next(mbq.physics);
    if k == 1
        % First window: initialise from preconditioned weights (optional)
        [~, ~, parameters] = NN_VlasovPoisson_dec(...
            X(1,:), X(2,:), X(3,:), 0, [], PINN.layer, Scaling);
        % Load pretrained initialisation if available
        if exist("preconditioned.mat", "file")
            load("preconditioned.mat");
            disp("Loaded preconditioned network weights.");
        end
    end
    % Verify network is functional with a forward pass
    [~, ~] = NN_VlasovPoisson_dec(...
        X(1,:), X(2,:), X(3,:), 1, parameters, PINN.layer, Scaling);
    clear X; reset(mbq.physics);

    %% Accelerated Model Gradient ------------------------------------------
    modelgradient = dlaccelerate(@modelgradient_twostream_dec);

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

    %% Output File Setup ---------------------------------------------------
    % Automatically increments filename to avoid overwriting previous runs.
    basename = "results\TwoStream_tdd_t" + geo.tn_range(1);
    counter = 1;
    filename = sprintf('%s%d.mat', basename, counter);
    while exist(filename, 'file')
        counter = counter + 1;
        filename = sprintf('%s%d.mat', basename, counter);
    end
    clear counter basename;

    %% Loss Weights Initialisation -----------------------------------------
    weights = PINN.weights;

    %% Training Loop -------------------------------------------------------

    epoch       = 0;
    convergence = false;
    LearningRate = eta_min;             % will be updated by schedule

    while ~convergence

        % parameters = dlupdate(@gpuArray,parameters);
        % dScaling = dlupdate(@gpuArray,Scaling);
        % dlayer = dlupdate(@gpuArray,PINN.layer);
        % dweights = dlupdate(@gpuArray,weights);
        layer = PINN.layer;

        tic;
        epoch = epoch + 1;

        %% Inner loop: gradient steps per epoch
        for i = 1 : PINN.iteration_per_epochs

            iteration = iteration + 1;

            % Sample fresh mini-batches (Sobol sequence, resampled each iteration)
            Xp = next(mbq.physics);     % physics collocation points
            Xg = next(mbq.gridX);       % spatial grid points (for conservation)
            Xi = next(mbq.initial);     % initial condition points

            % Evaluate loss and gradients
            [gradients, Loss, Losses] = dlfeval(modelgradient, ...
                parameters, layer, Scaling, ...
                Xp(1,:), Xp(2,:), Xp(3,:), ...
                Xi(1,:), Xi(2,:), Xi(3,:), Xi(4,:), Xi(5,:), ...
                Xg(1,:), Xg(2,:), Xg(3,:), wx, ...
                weights);

            % Parameter update
            if PINN.solver == "adam"
                [parameters, averageGrad, averageSqGrad] = adamupdate(...
                    parameters, gradients, averageGrad, averageSqGrad, ...
                    iteration, LearningRate);
            elseif PINN.solver == "rms"
                [parameters, averageSqGrad] = rmspropupdate(...
                    parameters, gradients, averageSqGrad, LearningRate);
            end

        end

        % Reset mini-batch queues for next epoch
        reset(mbq.physics);
        reset(mbq.gridX);
        reset(mbq.initial);

        %% Periodic Evaluation and Plotting --------------------------------

        if mod(epoch, Plot.rate) == 0

            % Plot individual weighted losses
            subplot(3,4,1); hold on;
            plot(epoch, Losses(1).*weights(1), '.r', 'MarkerSize', 16);
            plot(epoch, Losses(2).*weights(2), '.b', 'MarkerSize', 16);
            plot(epoch, Losses(3).*weights(3), '.g', 'MarkerSize', 16);
            plot(epoch, Losses(4).*weights(4), '.m', 'MarkerSize', 16);
            set(gca, 'YScale', 'log'); grid on; grid minor;
            title('Weighted losses'); legend('IC','Vlasov','Poisson','Reg');

            % Plot total loss
            subplot(3,4,5); hold on;
            plot(epoch, Loss, '.b', 'MarkerSize', 16);
            set(gca, 'YScale', 'log'); grid on; grid minor;
            title('Total loss');

            %%
            % Evaluate and plot PINN predictions at three time instants
            for p = 1:3
                Plot.dlX(3,:) = PIC.Results(ind_pic(p)).t;
                [f, E] = dlfeval(@NN_VlasovPoisson_Predict, ...
                    Plot.dlX(1,:), Plot.dlX(2,:), Plot.dlX(3,:), ...
                    parameters, PINN.layer, Scaling);

                f  = reshape(f, size(Plot.xn));
                E  = reshape(E, size(Plot.xn));
                fn = PIC.Results(ind_pic(p)).f;

                % Relative L2 error vs PIC reference
                RMSE(p, epoch) = norm(f(:) - fn(:)) / norm(fn(:));
                RMSE_E(p,epoch) = norm(E(1,:)'-PIC.Results(ind_pic(p)).E(:)) /...
                    norm(PIC.Results(ind_pic(p)).E(:)); 

                % Phase-space plot
                subplot(3, 4,  (p)*4-1);
                contourf(Plot.xn, Plot.vn, f, 30, 'LineStyle', 'none');
                colorbar(); colormap("jet"); ylim([-3 3]);
                grid on; grid minor;
                xlabel('x [λ_D]'); ylabel('v [v_{th}]');
                title("PINN " + titles(p) + " [t_0]");

                % Electric field profile
                subplot(3, 4, (p)*4);
                hold off;
                plot(PIC.PIC.geo.xn, PIC.Results(ind_pic(p)).E, 'b', ...
                    'DisplayName', 'PIC');
                hold on;
                plot(Plot.xn(1,:), E(1,:), 'r--', 'DisplayName', 'PINN');
                legend; grid on;
                title("E(x) at " + titles(p));

            end

            fprintf('  L2 errors: [%.4f, %.4f, %.4f]\n', ...
                RMSE(1,epoch), RMSE(2,epoch), RMSE(3,epoch));

            % Plot total loss
            subplot(3,4,9)
            plot(RMSE_E(:,1:epoch)', '.', 'MarkerSize', 16);
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

    %% Update Scaling for Next Window (based on current solution RMS) ------
    % This ensures the loss weighting remains appropriate as the
    % solution amplitude evolves through the nonlinear phase.
    [f_out, E_out, ~] = dlfeval(@NN_VlasovPoisson_Predict, ...
        Xp(1,:), Xp(2,:), Xp(3,:), parameters, PINN.layer, Scaling);
    new_scaling_f0 = sqrt(mean(f_out.^2));
    new_scaling_E0 = sqrt(mean(E_out.^2));

    fprintf('  Updated scaling: f0=%.4f, E0=%.4f\n', new_scaling_f0, new_scaling_E0);

    %% Save Results --------------------------------------------------------
    save(filename, 'Model', 'PINN', 'geo', 'Scaling', 'RMSE','RMSE_E');
    fprintf('  Results saved to: %s\n', filename);

end % for k = 1:Sim.N

fprintf('\n=== Training complete for all %d time windows ===\n', Sim.N);
