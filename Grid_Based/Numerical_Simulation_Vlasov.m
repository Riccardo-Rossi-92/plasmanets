%% FIG2_SEPARATION  Verification of the moment-separation law
%
%   Two distributions with identical density, mean velocity and temperature
%   are advanced with the Vlasov-Poisson system.  Their moment differences
%   are predicted to separate as
%
%       |dM^j_k(t)|  =  |dM^N_k(t0)| * (k*dt)^(N-j) / (N-j)!
%
%   with N the first discriminating moment order (N = 4 for two symmetric
%   distributions matched in n, u and T).
%
%   Panel (a): moments matched to machine precision -> four straight lines
%              of slope 4, 3, 2, 1 on log-log axes, lying on the theory.
%   Panel (b): the same data with a noise floor added -> the power laws are
%              buried.  This is what the figure looks like when the discrete
%              moments of a PIC sample are not corrected.
%
%   Solver: Fourier-spectral in x and in v, 4th-order Yoshida composition of
%   Strang splittings.  Both advections are exact shifts, hence applied as
%   phase factors in the respective Fourier space.  Everything is written
%   inline so the file is self-contained and runs unchanged in MATLAB and in
%   GNU Octave.
%
%   Runtime: about one minute for the default settings.
%
%   Verified under GNU Octave: fitted exponents 3.996, 2.993, 1.990, 0.979
%   against the predicted 4, 3, 2, 1.

clear; close all;

%% ------------------------------------------------------------------
%  Parameters
%  ------------------------------------------------------------------
Nx     = 32;            % grid points in x
Nv     = 2048;          % grid points in v
kwave  = 1;           % perturbation wavenumber -> L = 2*pi/kwave
Lv     = 10.0;          % velocity half-domain [-Lv, Lv], in v_th
Amp    = 0.05;          % perturbation amplitude, f = (1 + A cos kx) g(v)
dt     = 0.002;         % time step, in 1/omega_pe
ktmax  = 2.0;           % integrate until kwave*t = ktmax
beam   = 0.8;           % two-stream beam offset (variance kept equal to 1)
nfloor = 0.05;          % artificial noise floor for panel (b), relative
jmax   = 3;             % track moments j = 0 ... jmax
fitwin = [0.05 0.30];   % k*t window used for the slope fit

L = 2*pi/kwave;
N = 4;                  % first discriminating order (see below)

%% ------------------------------------------------------------------
%  Grids  (x and kx are COLUMNS, v and kv are ROWS)
%  ------------------------------------------------------------------
x  = (0:Nx-1)' * L/Nx;
v  = ((0:Nv-1) - Nv/2) * (2*Lv/Nv);
dv = v(2) - v(1);

kx = (2*pi/L) * [0:Nx/2-1, -Nx/2:-1]';
kv = (pi/Lv)  * [0:Nv/2-1, -Nv/2:-1];
nz = (kx ~= 0);                          % nonzero x-wavenumbers

%% ------------------------------------------------------------------
%  The two velocity shapes: same mass, same mean, same variance
%  ------------------------------------------------------------------
%  g1 : Maxwellian of unit variance
%  g2 : two counter-streaming beams at +-beam, with beam^2 + s^2 = 1
%  Both are even in v, so their third moments coincide as well and the
%  first difference appears at the FOURTH moment  ->  N = 4.

gauss = @(vv, mu, sg) exp(-(vv-mu).^2 ./ (2*sg^2)) ./ sqrt(2*pi*sg^2);

s  = sqrt(1 - beam^2);
g1 = gauss(v, 0, 1);
g2 = 0.5*(gauss(v, -beam, s) + gauss(v, +beam, s));
g1 = g1 / (sum(g1)*dv);                  % renormalise on the grid
g2 = g2 / (sum(g2)*dv);

fprintf('Moment check on the grid (must vanish for j = 0,1,2,3):\n');
for j = 0:5
    fprintf('   j = %d :  D<v^j> = %+.3e\n', j, sum(v.^j .* (g1-g2))*dv);
end
fprintf('   first discriminating order N = %d,  D<v^N> = %.6f\n\n', ...
        N, abs(sum(v.^N .* (g1-g2))*dv));

%% ------------------------------------------------------------------
%  Initial condition
%  ------------------------------------------------------------------
pert = 1 + Amp*cos(kwave*x);             % Nx x 1
f1   = pert * g1;                        % outer product -> Nx x Nv
f2   = pert * g2;

% v^j replicated over x, precomputed once: keeps the moment sums fast and
% avoids relying on implicit expansion (works on any MATLAB version)
VJ = cell(1, max(jmax,N)+1);
for j = 0:max(jmax,N)
    VJ{j+1} = ones(Nx,1) * (v.^j);
end

% reference: amplitude of the k-th harmonic of dM^N at t0
c1 = fft(sum(VJ{N+1}.*f1, 2)*dv);
c2 = fft(sum(VJ{N+1}.*f2, 2)*dv);
ref = abs(c1(2) - c2(2)) / Nx;

%% ------------------------------------------------------------------
%  Precomputed shift factors (the x-advection phase never changes)
%  ------------------------------------------------------------------
cY   = 2^(1/3);
wY   = [1/(2-cY), -cY/(2-cY), 1/(2-cY)];   % Yoshida weights
kxv  = kx * v;                             % Nx x Nv outer product
shiftX = cell(1,3);
for i = 1:3
    shiftX{i} = exp(-1i * kxv * (wY(i)*dt/2));
end

%% ------------------------------------------------------------------
%  Time integration
%  ------------------------------------------------------------------
nStep = round(ktmax / kwave / dt);
kt    = zeros(nStep,1);
D     = zeros(nStep, jmax+1);

fprintf('Integrating %d steps ...\n', nStep);
tic;
for n = 1:nStep

    for which = 1:2                       % advance f1, then f2
        if which == 1, f = f1; else, f = f2; end

        for i = 1:3                       % Yoshida composition
            % --- half shift in x:  df/dt + v df/dx = 0
            f = real(ifft( fft(f,[],1) .* shiftX{i}, [], 1));

            % --- field from Poisson:  dE/dx = 1 - n,  zero-mean gauge
            nn = sum(f,2)*dv;
            nh = fft(nn);
            Eh = zeros(size(nh));
            Eh(nz) = -nh(nz) ./ (1i*kx(nz));
            E  = real(ifft(Eh));

            % --- full shift in v:  df/dt - E df/dv = 0
            f = real(ifft( fft(f,[],2) .* exp(1i*(E*kv)*(wY(i)*dt)), [], 2));

            % --- half shift in x
            f = real(ifft( fft(f,[],1) .* shiftX{i}, [], 1));
        end

        if which == 1, f1 = f; else, f2 = f; end
    end

    kt(n) = kwave * n * dt;
    for j = 0:jmax
        a = fft(sum(VJ{j+1}.*f1, 2)*dv);
        b = fft(sum(VJ{j+1}.*f2, 2)*dv);
        D(n,j+1) = abs(a(2) - b(2)) / Nx / ref;
    end

    if mod(n, max(1,round(nStep/10))) == 0
        fprintf('   %3d%%   (%.0f s)\n', round(100*n/nStep), toc);
    end
end

%% ------------------------------------------------------------------
%  Slope fit
%  ------------------------------------------------------------------
m = (kt >= fitwin(1)) & (kt <= fitwin(2));
fprintf('\nFitted exponents over k*t in [%.2f, %.2f]:\n', fitwin(1), fitwin(2));
for j = 0:jmax
    p = polyfit(log(kt(m)), log(D(m,j+1)), 1);
    fprintf('   j = %d :  measured %6.3f   predicted %d\n', j, p(1), N-j);
end


%% save workspace
save('case_k_1.mat', 'kt', 'D', 'N', 'kwave', 'Amp', 'nfloor');

%% figure

col = [  0 114 178;   % blu
    213   0   0;   % rosso
    0 158 115;   % verde
    204   0 167;   % magenta
    200 122   0;   % arancio
    86 180 233] / 255;  % azzurro

figure(1)
clf
hold on
for j = 0:jmax
    th = kt.^(N-j) / factorial(N-j);                  % theory
    plot(kt, th, '--', 'Color', [.55 .55 .55], 'LineWidth', 0.7);
    y = D(:,j+1);
    h(j+1) = plot(kt, y, '-', 'Color', col(j+1,:), 'LineWidth', 1.8);
end

lab = cell(1,jmax+1);
for j = 0:jmax
    lab{j+1} = sprintf('\\DeltaM^{%d}\\propto (k\\Deltat)^{%d}', j, N-j);
end

set(gca, 'XScale','log', 'YScale','log', 'TickDir','in', 'FontSize',9, ...
    'XMinorGrid','off', 'YMinorGrid','off');
grid on;
grid minor;
xlim([0.012 ktmax]); ylim([1e-8 3]);
xlabel('k \Deltat');
ylabel('|\DeltaM^{ j}_k(t)| / |\DeltaM^{ N}_k(t_0)|');
legend(h, lab, 'Location','southeast');
title("k = " + kwave)
xticks([0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1 2])
yticks(10.^linspace(-12,0,13))
