function [mbq,wx,PINN] = prepare_domain_twostream_inverse_u_T(PINN,geo,Config,Model)
% PREPARE_DOMAIN_TWOSTREAM_INVERSE_U_T  Training set, enriched measurements.
%
%   As prepare_domain_twostream_inverse, with momentum and temperature added
%   to the measurement records.


if isempty(Model)
    first_time_slice = 1;
else
    first_time_slice = 0;
end

%% Equilibrium condition
vth = 0.2; vb = 1;

%% Structured Grid on x with v to 5 for measurements second orders

PINN.gridX.Nx = 50;
PINN.gridX.Nvt = 20;
PINN.gridX.batch = PINN.gridX.Nx.*PINN.gridX.Nvt;

x = linspace(geo.xn_range(1),geo.xn_range(2),PINN.gridX.Nx);
PINN.gridX.dx = x(2)-x(1);

p = sobolset(1);
t = p(1:PINN.gridX.Nvt*PINN.iteration_per_epochs,1);

v = ones(size(t)).*geo.vn_range(2);
t = t.*(geo.tn_range(2) - geo.tn_range(1)) + geo.tn_range(1);

xt = []; vt = []; tt = [];

for i = 1 : PINN.iteration_per_epochs*PINN.gridX.Nvt
    xt = [xt, x];
    vt = [vt, ones(size(x)).*v(i)];
    tt = [tt, ones(size(x)).*t(i)];
end

x = xt;
v = vt;
t = tt;
class = x.*0+1;
class_pl = double(x<geo.xn_range(1)+0.01);
class_pr = double(x>geo.xn_range(2)-0.01);

wx = zeros(PINN.gridX.Nx*PINN.gridX.Nvt,PINN.gridX.Nvt);

for i = 1 : PINN.gridX.Nvt

    wx((i-1)*PINN.gridX.Nx+1:i*PINN.gridX.Nx,i) = 1;

end

wx = wx';

clear xt vt tt

ds = arrayDatastore([x; v; t; class; class_pl; class_pr]');

mbq.gridX = minibatchqueue(ds,...
    "MiniBatchFormat","BC",...
    "OutputEnvironment",Config.environment,...
    "MiniBatchSize",PINN.gridX.batch);

clear points ds

%% Collocation points
PINN.collocation_points = PINN.iteration_per_epochs*PINN.physics_mini_batch;

% generate points (dimless domain)
points = sobolset(3);
points = points(1:PINN.collocation_points,:);
points(:,1) = points(:,1).*(geo.xn_range(2) - geo.xn_range(1)) + geo.xn_range(1);
points(:,2) = points(:,2).*(geo.vn_range(2) - geo.vn_range(1)) + geo.vn_range(1);
points(:,3) = points(:,3).*(geo.tn_range(2) - geo.tn_range(1)) + geo.tn_range(1);

% supervised part (just for testing architecture and model gradient)

load("PIC\pic_twostream.mat","Results","PIC"); Geo = PIC.geo; clear PIC

for l = 1 : length(Results)
    Epic(:,l) = Results(l).E;
    M(:,:,l) = Results(l).f;
    tn(l) = Results(l).t;
end

[xg,vg,tg] = meshgrid(Geo.xn,Geo.vn,tn);
fs = interp3(xg,vg,tg,M,points(:,1),points(:,2),points(:,3));

[xg,tg] = meshgrid(Geo.xn,tn);
E = interp2(xg,tg,Epic',points(:,1),points(:,3));

points(:,4) = fs;
points(:,5) = E;

vt = points(:,2); 

clear xg vg tg fs

ds = arrayDatastore(points);

mbq.physics = minibatchqueue(ds,...
    "MiniBatchFormat","BC",...
    "OutputEnvironment",Config.environment,...
    "MiniBatchSize",PINN.physics_mini_batch);

clear points ds

class = zeros(1,PINN.physics_mini_batch);

%% initial condition

points = sobolset(2);
points = points(1:PINN.boundary_mini_batch*PINN.iteration_per_epochs,:);
x = points(:,1).*(geo.xn_range(2) - geo.xn_range(1)) + geo.xn_range(1);
v = points(:,2).*(geo.vn_range(2) - geo.vn_range(1)) + geo.vn_range(1);
t0 = geo.tn_range(1);
t = ones(size(v)).*t0;

f0 = 1./(sqrt(2*pi).*vth).*exp(-((v-vb)./vth).^2/2) + 1./(sqrt(2*pi).*vth).*exp(-((v+vb)./vth).^2/2);
f0 = f0/2;

if Config.PIC_init == 1
    [~,indt] = min(abs(tn-t0));

    E = interp1(Geo.xn,Epic(:,indt),x);
    f = interp2(Geo.xn,Geo.vn,M(:,:,indt),x,v)-f0;
else
    if first_time_slice == 1
        
        k = 1;
        p0 = 0.1;

        f = f0.*(1+p0.*cos(k.*x));
        E = -p0/k*sin(k.*x);

    else
        last_model = length(Model);
        [f,E] = dlfeval(@NN_VlasovPoisson_Predict,...
            dlarray(x','CB'),...
            dlarray(v','CB'),...
            dlarray(t','CB'),...
            Model{last_model}.parameters,...
            Model{last_model}.PINN.layer,...
            Model{last_model}.Scaling);
        f = f'; E = E';
    end
end

ds = arrayDatastore([x,v,t,f,E,ones(size(E))*3]);

mbq.initial = minibatchqueue(ds,...
    "MiniBatchFormat","BC",...
    "OutputEnvironment",Config.environment,...
    "MiniBatchSize",PINN.boundary_mini_batch);

%% Measurements - first order
clear nm Em tm xm um Tm
load("PIC\pic_twostream.mat","Results","PIC"); 

xm = PIC.geo.xn;

for i = 1 : length(Results)
    tm(i) = Results(i).t;
    Em(i,:) = Results(i).E;
    nm(i,:) = sum(Results(i).f,1)*PIC.geo.dv;
    um(i,:) = Results(i).u;
    Tm(i,:) = Results(i).T;
end

[xm,tm] = meshgrid(xm,tm);

tm = tm(:);
xm = xm(:);
Em = Em(:);
nm = nm(:);
um = um(:);
Tm = Tm(:);

ind = find(tm>=geo.tn_range(1)&tm<=geo.tn_range(2));

tm = tm(ind);
xm = xm(ind);
vm = (xm*0+1).*geo.vn_range(2);
Em = Em(ind);
nm = nm(ind);
um = um(ind);
Tm = Tm(ind);

ds = arrayDatastore([xm,vm,tm,nm,Em,um,Tm,ones(size(Em))*7]);

mbq.measurements = minibatchqueue(ds,...
    "MiniBatchFormat","BC",...
    "OutputEnvironment",Config.environment,...
    "MiniBatchSize",PINN.measurements_mini_batch);

shuffle(mbq.measurements)

%%

% dlClass = dlarray(class,'CB');
wx = dlarray(wx,'CB');
% wv = dlarray(wv,'CB');

% GPU array
if Config.GPU == 1
    wx = gpuArray(wx);
    % wv = gpuArray(wv);
end

end