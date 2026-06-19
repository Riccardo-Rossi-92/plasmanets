function [gradients, Loss, Losses] = modelgradient_inverse_u_T(...
                                        parameters,PINN,Scaling,...
                                        xp,vp,tp,...
                                        xm,vm,tm,nm,Em,um,Tm,...
                                        xg,vg,tg,wgx,...
                                        alphas)

%% Predict
[K, M] = NN_VlasovPoisson_dec_u_T(xp,vp,tp, 1, parameters, PINN.layer,Scaling);

F = K(1,:);
Fv = K(2,:);
FT = K(3,:);
E = M(1,:);

f = dlgradient(sum(F,'all'),vp,"EnableHigherDerivatives",true);
gradf = dlgradient(sum(f,'all'),{xp,vp,tp});

dfdx = gradf{1};
dfdv = gradf{2};
dfdt = gradf{3};

dEdx = dlgradient(sum(E,'all'),xp);

fv = dlgradient(sum(Fv,'all'),vp,"EnableHigherDerivatives",true);
fT = dlgradient(sum(FT,'all'),vp,"EnableHigherDerivatives",true);

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

%% Consistency

Loss_cons = mean((f.*vp-fv).^2)./vinf.^2 + mean((f.*vp.^2-fT).^2)./vinf.^4; 

%% Loss Poisson
K = NN_VlasovPoisson_dec_u_T(xp,vp*0+vinf,tp, 1, parameters, PINN.layer, Scaling);
F = K(1,:);
eq = dEdx + F - 1;
Loss_Poisson = mean(eq.^2)./sE.^2;

%% F to 0 for v to -inf
% F1 = NN_VlasovPoisson_dec_m3(xp,vp*0-vinf,tp, 1, parameters, PINN.layer, Scaling);
% Loss_cdf0 = mean(F1.^2)./sE.^2;

%% Conservation of mass anc charge
% Loss_conservation = mean(E).^2./sE.^2;
          
%% Conservation of charge and mass

[K,E] = NN_VlasovPoisson_dec_u_T(xg,vg,tg, 1, parameters, PINN.layer, Scaling);

F = K(1,:);

E_mean = sum(E.*wgx,2)./sum(wgx,2);
F_mean = sum(F.*wgx,2)./sum(wgx,2)-1;
Loss_conservation = mean(E_mean.^2)./sE.^2 + mean(F_mean.^2)./sE.^2;

%% Measurements

[K,E] = NN_VlasovPoisson_dec_u_T(xm,vm,tm, 1, parameters, PINN.layer, Scaling);

F = K(1,:);
Fv = K(2,:);
FT = K(3,:);

sigEm = std(Em);
signm = std(nm);
sigum = std(um);
sigTm = std(Tm);

Z2E = (E-Em).^2./sigEm.^2;
Z2n = (F-nm).^2./signm.^2;
Z2u = (Fv-um).^2./sigum.^2;
Z2T = (FT-Tm).^2./sigTm.^2;

E_mean = mean(tanh(Z2E).*Z2E);
F_mean = mean(tanh(Z2n).*Z2n);
U_mean = mean(tanh(Z2u).*Z2u);
T_mean = mean(tanh(Z2T).*Z2T);

Loss_measurements = E_mean + F_mean + U_mean + T_mean;

%% All losses
Losses = [Loss_measurements;...
            Loss_Vlasov ;...
            Loss_Poisson;...
            Loss_conservation + Loss_NNeg;...
            Loss_cons;...
            Loss_NNeg];

Loss = sum(alphas.*Losses);

gradients = dlgradient(Loss,parameters);

end