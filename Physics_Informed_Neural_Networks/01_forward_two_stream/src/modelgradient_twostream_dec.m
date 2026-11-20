function [gradients, Loss, Losses] = modelgradient_twostream_dec(...
                                        parameters,layer,Scaling,...
                                        xp,vp,tp,...
                                        xi,vi,ti,f_ic,E_ic,...
                                        xg,vg,tg,wgx,...
                                        alphas)

% MODELGRADIENT_TWOSTREAM_DEC  Loss and gradients for the forward problem.
%
%   Assembles five residuals, each normalised by the scale of the quantity it
%   constrains, and returns their weighted sum (weights alphas) together with
%   the gradients with respect to parameters:
%
%       Loss_IC            initial condition on f and E
%       Loss_Vlasov        df/dt + v df/dx - E df/dv = 0 at collocation points
%       Loss_Poisson       dE/dx + F(v->inf) - 1 = 0
%       Loss_conservation  zero mean of E and of the density over the domain
%       Loss_NNeg          penalty on any negative f


%% Predict
[K, M] = NN_VlasovPoisson_dec(xp,vp,tp, 1, parameters, layer,Scaling);

F = K(1,:);
E = M(1,:);

f = dlgradient(sum(F,'all'),vp,"EnableHigherDerivatives",true);
gradf = dlgradient(sum(f,'all'),{xp,vp,tp});

dfdx = gradf{1};
dfdv = gradf{2};
dfdt = gradf{3};

dEdx = dlgradient(sum(E,'all'),xp);

%% Scaling

vinf = 5;
sE = std(E_ic);
sf = std(f_ic);
vth = 1;

%% Loss Vlasov
eq = dfdt + vp.*dfdx - E.*dfdv;
Loss_Vlasov = mean(eq.^2).*(vth./(sE)).^2;

%% Non Negativity
Loss_NNeg = mean(relu(-f).^2)./sf.^2;

%% Loss Poisson
F = NN_VlasovPoisson_dec(xp,vp*0+vinf,tp, 1, parameters, layer, Scaling);
eq = dEdx + F - 1;
Loss_Poisson = mean(eq.^2)./sE.^2;
          
%% Initial Condition

[F,E] = NN_VlasovPoisson_dec(xi,vi,ti, 1, parameters, layer, Scaling);

f = dlgradient(sum(F),vi);

sfi = std(f_ic)*0.1;
sEi = std(E_ic);

sf2eq = (1./sfi).^2+(1./sE).^2;
sE2eq = (1./sEi).^2+(1./sE).^2;

Zf = ((f-f_ic).^2).*sf2eq;
ZE = ((E-E_ic).^2).*sE2eq;

Loss_IC = mean(Zf) + mean(ZE);

%% Conservation of charge and mass

[F,E] = NN_VlasovPoisson_dec(xg,vg,tg, 1, parameters, layer, Scaling);

E_mean = sum(E.*wgx,2)./sum(wgx,2);
F_mean = sum(F.*wgx,2)./sum(wgx,2)-1;
Loss_conservation = mean(E_mean.^2)./sE.^2 + mean(F_mean.^2)./sE.^2;

%% All losses
Losses = [Loss_IC;...
            Loss_Vlasov;...
            Loss_Poisson;...
            Loss_conservation;...
            Loss_NNeg];

Loss = sum(alphas.*Losses);

gradients = dlgradient(Loss,parameters);

end