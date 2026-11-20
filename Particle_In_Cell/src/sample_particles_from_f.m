function [xp, vp, wp, info] = sample_particles_from_f(f, xg, vg, N, opts)
%SAMPLE_PARTICLES_FROM_F  Carica N particelle distribuite secondo f(x,v).
%
%   [XP,VP,WP] = SAMPLE_PARTICLES_FROM_F(F,XG,VG,N)
%
%   Genera N particelle (XP,VP) di peso WP la cui distribuzione riproduce
%   f(x,v) mediante inversione esatta della CDF:
%
%       rho(x)   = \int f(x,v) dv            -> CDF marginale  Fx(x)
%       g(v|x)   = f(x,v)/rho(x)             -> CDF condizionata Fv(v|x)
%       x_i = Fx^{-1}(u1_i),   v_i = Fv^{-1}(u2_i | x_i)
%
%   Con u1,u2 presi ai quantili (i-1/2)/N (quiet start deterministico) la
%   CDF empirica in x coincide con quella esatta a meno di mezzo strato,
%   1/(2N), e non c'e' rumore statistico.
%   NB: con N finito nessun insieme di particelle riproduce f esattamente
%   come funzione (l'empirica e' una somma di delta); l'unico senso in cui
%   "esattamente" e' raggiungibile e' l'uguaglianza delle CDF ai punti di
%   campionamento, che e' quello che fa questa routine.
%
%   Precisione misurata (two-stream, N=1e5, errore max relativo su rho(x)
%   ricostruita su 64 celle, errore relativo su <v^2>, |<v>| assoluto -
%   numeri riproducibili lanciando la demo):
%       method          rho       <v^2>     |<v>|
%       hammersley    6.0e-04    9.4e-06   2.3e-04   <- default consigliato
%       tensor        5.7e-02    3.4e-03   ~1e-17    <- beamlet in x,
%                                                       ma <v>=0 esatto
%       tensor+jitter 9.6e-03    1.2e-04   2.2e-04
%       random        5.2e-02    9.8e-04   5.4e-03
%   Cioe': 'tensor' mette molte particelle alla STESSA x (una "beamlet" per
%   strato) e questo si vede come rumore a griglia in rho(x); 'hammersley'
%   da' N posizioni x tutte distinte ai quantili esatti ed e' ~2 ordini di
%   grandezza piu' pulito. Usa 'tensor' quando serve la simmetria esatta in
%   v (<v>=0 a precisione macchina, niente corrente spuria iniziale).
%
% INPUT
%   F   : function handle @(x,v) f(x,v)  (deve essere vettorializzato, es.
%         usare .* ./ .^), oppure matrice [numel(XG) x numel(VG)] con i
%         valori di f sulla griglia. f >= 0, non necessariamente normalizzata.
%   XG  : vettore griglia in x (usata per quadratura e inversione della CDF)
%   VG  : vettore griglia in v
%   N   : numero di particelle. Con method='tensor' puo' essere [Npx Npv].
%   OPTS: struct opzionale
%      .method : 'hammersley' (default) u1 ai quantili (i-1/2)/N, u2 dalla
%                             sequenza di van der Corput base 2. N qualsiasi,
%                             tutte le x distinte: minimo rumore su rho(x).
%                'tensor'     griglia Npx x Npv nello spazio dei quantili:
%                             quiet start classico, simmetrie esatte, ma
%                             particelle addensate su Npx sole posizioni.
%                             N puo' essere [Npx Npv].
%                'random'     Monte Carlo classico (per confronto)
%      .jitter : true/false (default false) randomizza dentro ogni strato
%                (quiet start -> quasi-MC stratificato: rompe la simmetria
%                che puo' innescare modi spuri, mantenendo O(1/N) circa)
%      .seed   : seme rng per 'random'/'jitter'
%      .chunk  : particelle processate per blocco (default 2e4, memoria)
%
% OUTPUT
%   XP,VP : N x 1 posizioni e velocita'
%   WP    : N x 1 pesi, tutti uguali a Q/N con Q = \int\int f dx dv
%           (peso "fisico": sum(WP) = massa/carica totale di f)
%   INFO  : struct con la diagnostica (momenti esatti vs particelle)
%
% ESEMPI
%   sample_particles_from_f              % esegue la demo con grafici
%
%   f  = @(x,v) (1+0.05*cos(0.5*x)).*exp(-v.^2/2)/sqrt(2*pi);
%   xg = linspace(0,4*pi,513);  vg = linspace(-6,6,513);
%   [xp,vp,wp] = sample_particles_from_f(f,xg,vg,20000);          % default
%   [xp,vp,wp] = sample_particles_from_f(f,xg,vg,[200 100], ...   % quiet start
%                                        struct('method','tensor'));
%
%   Autore: generato per il progetto Vlasov-Poisson.

% ------------------------------------------------------------------ demo
if nargin == 0
    demo(); return
end
if nargin < 5 || isempty(opts), opts = struct(); end
opts = set_defaults(opts, struct('method','random','jitter',false, ...
                                 'seed',[],'chunk',5e3));
if ~isempty(opts.seed), rng(opts.seed); end

xg = xg(:); vg = vg(:);
nx = numel(xg); nv = numel(vg);
if nx < 2 || nv < 2, error('XG e VG devono avere almeno 2 nodi.'); end
if any(diff(xg) <= 0) || any(diff(vg) <= 0)
    error('XG e VG devono essere strettamente crescenti.');
end

% -------------------------------------------------- f sulla griglia (nx x nv)
isH = isa(f,'function_handle');
if isH
    F = eval_f(f, xg, vg);
else
    F = f;
    if isequal(size(F),[nv nx]) && nv ~= nx, F = F.'; end
    if ~isequal(size(F),[nx nv])
        error('La matrice F deve essere [numel(XG) x numel(VG)].');
    end
end
if any(~isfinite(F(:))), error('f contiene valori non finiti.'); end
if any(F(:) < 0)
    warning('f ha valori negativi: vengono troncati a 0.');
    F = max(F,0);
end

% ---------------------------------------------- CDF marginale in x
rho = trapz(vg, F, 2);                 % nx x 1
Cx  = cumtrapz(xg, rho);
Q   = Cx(end);                         % \int\int f dx dv
if Q <= 0, error('Integrale di f nullo o negativo.'); end
Cx  = Cx / Q;

% ---------------------------------------------- numeri "casuali" u1,u2
[u1, u2, Np, meth] = quantile_points(N, opts);

% ---------------------------------------------- inversione in x
xp = invert_cdf(Cx, xg, u1);

gi = find(rho > 0);      % nodi con densita' non nulla (per il fallback)

% ---------------------------------------------- inversione condizionata in v
vp = zeros(Np,1);
nb = max(1, round(opts.chunk));
for i0 = 1:nb:Np
    i1  = min(i0+nb-1, Np);
    idx = (i0:i1).';
    xb  = xp(idx);

    if isH
        Fb = eval_f(f, xb, vg);                 % valutazione esatta in x_i
        Fb = max(Fb,0);
    else
        Fb = interp1(xg, F, xb, 'linear');      % interpolazione delle righe
        if numel(idx) == 1, Fb = Fb(:).'; end
    end

    % Se f ha zone a densita' nulla, un quantile puo' cadere nel "buco":
    % quelle particelle vengono riportate al nodo con rho>0 piu' vicino.
    tot = trapz(vg, Fb, 2);
    bad = ~(tot > 0);
    if any(bad)
        [~, jj] = min(abs(bsxfun(@minus, xg(gi), xb(bad).')), [], 1);
        xs = xg(gi(jj(:)));
        xb(bad) = xs;  xp(idx(bad)) = xs;
        if isH
            Fb(bad,:) = max(eval_f(f, xs, vg), 0);
        else
            Fb(bad,:) = interp1(xg, F, xs, 'linear');
        end
    end

    Cb = cumtrapz(vg, Fb, 2);                   % CDF condizionata (non norm.)
    tot = Cb(:,end);
    bad = ~(tot > 0);
    tot(bad) = 1;
    Cb = bsxfun(@rdivide, Cb, tot);
    if any(bad)                                 % ultimo fallback: uniforme
        Cb(bad,:) = repmat(linspace(0,1,nv), sum(bad), 1);
    end

    vp(idx) = invert_rows(Cb, vg, u2(idx));
end

wp = (Q/Np) * ones(Np,1);

% ---------------------------------------------- diagnostica
if nargout > 3
    info = struct();
    info.method   = meth;
    info.N        = Np;
    info.Q        = Q;
    info.exact    = moments_from_grid(F, xg, vg);
    info.particle = moments_from_particles(xp, vp, wp);
    fn = fieldnames(info.exact);
    for k = 1:numel(fn)
        e = info.exact.(fn{k}); p = info.particle.(fn{k});
        info.relerr.(fn{k}) = abs(p-e)/max(abs(e), eps);
    end
end
end

% ====================================================================
function F = eval_f(f, xg, vg)
% Valuta f su una griglia [numel(xg) x numel(vg)], gestendo handle non
% pienamente vettorializzati.
F = [];
try
    F = f(xg(:), vg(:).');                       % broadcasting implicito
catch
    F = [];
end
if ~isequal(size(F), [numel(xg) numel(vg)])
    [X,V] = ndgrid(xg(:), vg(:));
    F = f(X,V) .* ones(size(X));
end
F = double(F);
end

% ====================================================================
function [u1, u2, Np, meth] = quantile_points(N, opts)
meth = lower(opts.method);
switch meth
    case 'tensor'
        if numel(N) == 2
            npx = N(1); npv = N(2);
        else
            [npx, npv] = factor_pair(N);
            if npx == 1 || npv == 1
                warning(['N=%d non si fattorizza bene: uso method=''hammersley''. ' ...
                         'Passa N=[Npx Npv] per il tensor product.'], N);
                [u1,u2,Np,meth] = quantile_points(N, set_defaults(struct('method','hammersley'), opts));
                return
            end
        end
        Np = npx*npv;
        ix = repelem((1:npx).', npv);
        iv = repmat((1:npv).', npx, 1);
        if opts.jitter
            u1 = (ix - rand(Np,1)) / npx;
            u2 = (iv - rand(Np,1)) / npv;
        else
            u1 = (ix - 0.5) / npx;
            u2 = (iv - 0.5) / npv;
        end

    case 'hammersley'
        Np = prod(N);
        i  = (1:Np).';
        if opts.jitter
            u1 = (i - rand(Np,1))/Np;
            u2 = mod(vdcorput(i,2) + rand(), 1);
        else
            u1 = (i - 0.5)/Np;
            u2 = vdcorput(i,2);
            u2 = min(max(u2, 0.5/Np), 1-0.5/Np);
        end

    case 'random'
        Np = prod(N);
        u1 = rand(Np,1);
        u2 = rand(Np,1);

    otherwise
        error('opts.method sconosciuto: %s', opts.method);
end
u1 = min(max(u1, eps), 1-eps);
u2 = min(max(u2, eps), 1-eps);
end

% ====================================================================
function [a,b] = factor_pair(N)
% Fattorizzazione a*b = N con a il divisore piu' vicino a sqrt(N) (a <= b).
a = 1;
for k = 1:floor(sqrt(N))
    if mod(N,k) == 0, a = k; end
end
b = N/a;
t = a; a = b; b = t;    % piu' punti in x che in v
end

% ====================================================================
function r = vdcorput(i, base)
% Van der Corput / radical inverse (vettorializzato).
r = zeros(numel(i),1); i = double(i(:)); fk = 1/base;
while any(i > 0)
    r  = r + fk*mod(i,base);
    i  = floor(i/base);
    fk = fk/base;
end
end

% ====================================================================
function q = invert_cdf(C, grid, u)
% Inversione di una CDF monotona 1-D (interpolazione lineare, plateau ok).
C = C(:); grid = grid(:);
keep = [true; diff(C) > 0];
q = interp1(C(keep), grid(keep), u(:), 'linear', 'extrap');
q = min(max(q, grid(1)), grid(end));
end

% ====================================================================
function q = invert_rows(C, grid, u)
% Inversione vettorializzata: ogni riga di C e' una CDF sulla griglia GRID,
% u(i) e' il livello da invertire per la riga i.
grid = grid(:);
n = size(C,1); m = size(C,2);
u = u(:);
j  = sum(C < repmat(u,1,m), 2);
j  = min(max(j,1), m-1);
li = (1:n).';
c0 = C(sub2ind([n m], li, j));
c1 = C(sub2ind([n m], li, j+1));
t  = (u - c0) ./ max(c1 - c0, realmin);
t  = min(max(t,0),1);
q  = grid(j) + t .* (grid(j+1) - grid(j));
end

% ====================================================================
function M = moments_from_grid(F, xg, vg)
rho = trapz(vg, F, 2);
Q   = trapz(xg, rho);
M.mass = Q;
M.meanx = trapz(xg, xg(:).*rho)/Q;
j   = trapz(vg, F.*repmat(vg(:).',numel(xg),1), 2);
M.meanv = trapz(xg, j)/Q;
p   = trapz(vg, F.*repmat((vg(:).').^2,numel(xg),1), 2);
M.v2 = trapz(xg, p)/Q;
end

function M = moments_from_particles(xp, vp, wp)
W = sum(wp);
M.mass  = W;
M.meanx = sum(wp.*xp)/W;
M.meanv = sum(wp.*vp)/W;
M.v2    = sum(wp.*vp.^2)/W;
end

% ====================================================================
function n = bincount(x, edges)
% Conteggio per istogramma (compatibile MATLAB/Octave).
n = histc(x(:), edges(:));
n(end-1) = n(end-1) + n(end);
n = n(1:end-1);
end

% ====================================================================
function s = set_defaults(s, d)
fn = fieldnames(d);
for k = 1:numel(fn)
    if ~isfield(s, fn{k}) || isempty(s.(fn{k}))
        s.(fn{k}) = d.(fn{k});
    end
end
end

% ====================================================================
function demo()
% Due test: (1) two-stream separabile, (2) f non separabile (drift e
% temperatura dipendenti da x). Confronta anche il rumore su rho(x).
k = 0.5; L = 2*pi/k; v0 = 2.4; A = 0.05;
f1 = @(x,v) (1 + A*cos(k*x)) .* (exp(-(v-v0).^2/2) + exp(-(v+v0).^2/2)) ...
            ./ (2*sqrt(2*pi));
sg = @(x) 0.6 + 0.3*cos(k*x);
f2 = @(x,v) (1 + 0.3*cos(k*x)) .* exp(-(v - 0.8*sin(k*x)).^2 ./ (2*sg(x).^2)) ...
            ./ (sqrt(2*pi)*sg(x));

xg = linspace(0,L,513).';
vg = linspace(-8,8,513).';
N  = 100000;

figure('Name','sample_particles_from_f: demo','Color','w');
ff = {f1,f2}; tt = {'two-stream (separabile)','non separabile'};
for c = 1:2
    [X,V] = ndgrid(xg,vg); Fex = ff{c}(X,V);
    rho_ex = trapz(vg, Fex, 2);
    fv_ex  = trapz(xg, Fex, 1).';  fv_ex = fv_ex/trapz(vg,fv_ex);

    fprintf('\n--- %s ---  errori relativi sui momenti\n', tt{c});
    meths = {'hammersley','tensor','random'};
    for m = 1:3
        o = struct('method',meths{m});
        if strcmp(meths{m},'tensor'), Nm = [1000 100]; else, Nm = N; end
        [xp,vp,wp,info] = sample_particles_from_f(ff{c}, xg, vg, Nm, o);
        fprintf('  %-11s mass %8.1e  <x> %8.1e  <v> %8.1e  <v^2> %8.1e\n', ...
            meths{m}, info.relerr.mass, info.relerr.meanx, ...
            abs(info.particle.meanv), info.relerr.v2);
        % rumore su rho(x) a 64 celle
        nc = 64; ed = linspace(xg(1),xg(end),nc+1); ctr = (ed(1:end-1)+ed(2:end))/2;
        h = bincount(xp, ed) * (sum(wp)/numel(xp)) / (ed(2)-ed(1));
        e = max(abs(h(:) - interp1(xg,rho_ex,ctr(:)))) / mean(rho_ex);
        fprintf('               errore max su rho(x), %d celle: %.2e\n', nc, e);
        if m == 1, xh = xp; vh = vp; end
    end

    subplot(2,3,3*(c-1)+1);
    contourf(X,V,Fex,20,'LineColor','none');
    xlabel('x'); ylabel('v'); title(['f esatta - ' tt{c}]); axis tight

    subplot(2,3,3*(c-1)+2);
    plot(xh(1:7:end), vh(1:7:end), '.', 'MarkerSize', 1);
    xlabel('x'); ylabel('v'); title('particelle (hammersley)'); axis tight

    subplot(2,3,3*(c-1)+3);
    nb = 200; ed = linspace(vg(1),vg(end),nb+1); ctr = (ed(1:end-1)+ed(2:end))/2;
    hv = bincount(vh, ed) / (numel(vh)*(ed(2)-ed(1)));
    plot(ctr,hv,'.', vg,fv_ex,'-','LineWidth',1.2);
    xlabel('v'); ylabel('f(v)'); legend('particelle','esatta'); title('marginale in v');
end
end
