function [gradients, Loss, Losses] = modelgradient_inverse(...
                                        parameters,PINN,Scaling,...
                                        xp,vp,tp,...
                                        xm,vm,tm,nm,Em,...
                                        xg,vg,tg,wgx,...
                                        alphas)

% MODELGRADIENT_INVERSE  Loss and gradients for the inverse problem.
%
%   The initial condition is unknown and is replaced by a data term on the
%   measurements taken over the observation window.  Residuals: measurements
%   (density and electric field), Vlasov, Poisson, zero mean of E and of the
%   density, and a penalty on any negative f.  The measurement term uses
%   z^2*tanh(z^2) so that outliers saturate instead of dominating the loss.

%% Predict
[K, M] = NN_VlasovPoisson_dec(xp,vp,tp, 1, parameters, PINN.layer,Scaling);

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
sE = std(E);
sf = std(f);
vth = 1;

%% Loss Vlasov
eq = dfdt + vp.*dfdx - E.*dfdv;
Loss_Vlasov = mean(eq.^2).*(vth./sE).^2;

%% Non Negativity
Loss_NNeg = mean(relu(-f).^2)./sf.^2;

%% Loss Poisson
F = NN_VlasovPoisson_dec(xp,vp*0+vinf,tp, 1, parameters, PINN.layer, Scaling);
eq = dEdx + F - 1;
Loss_Poisson = mean(eq.^2)./sE.^2;

%% Conservation of charge and mass

[F,E] = NN_VlasovPoisson_dec(xg,vg,tg, 1, parameters, PINN.layer, Scaling);

E_mean = sum(E.*wgx,2)./sum(wgx,2);
F_mean = sum(F.*wgx,2)./sum(wgx,2)-1;
Loss_conservation = mean(E_mean.^2)./sE.^2 + mean(F_mean.^2)./sE.^2;

%% Measurements

[F,E] = NN_VlasovPoisson_dec(xm,vm,tm, 1, parameters, PINN.layer, Scaling);

sigEm = std(Em);
signm = std(nm);

Z2E = (E-Em).^2./sigEm.^2;
Z2n = (F-nm).^2./signm.^2;

E_mean = mean(tanh(Z2E).*Z2E);
F_mean = mean(tanh(Z2n).*Z2n);

Loss_measurements = E_mean + F_mean;

%% All losses
Losses = [Loss_measurements;...
            Loss_Vlasov ;...
            Loss_Poisson;...
            Loss_conservation; ...
            Loss_NNeg];

Loss = sum(alphas.*Losses);

gradients = dlgradient(Loss,parameters);

end