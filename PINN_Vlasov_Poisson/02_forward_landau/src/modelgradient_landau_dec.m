function [gradients, Loss, Losses] = modelgradient_landau_dec(...
                                        parameters,PINN,Scaling,...
                                        xp,vp,tp,df0dv,...
                                        xi,vi,ti,f1_ic,E1_ic,...
                                        xg,vg,tg,wgx,...
                                        alphas)

%% Predict
[K, M] = NN_VlasovPoisson_dec(xp,vp,tp, 1, parameters, PINN.layer,Scaling);

F1 = K(1,:);
E1 = M(1,:);

f1 = dlgradient(sum(F1,'all'),vp,"EnableHigherDerivatives",true);
gradf = dlgradient(sum(f1,'all'),{xp,vp,tp});

df1dx = gradf{1};
df1dv = gradf{2};
df1dt = gradf{3};

dE1dx = dlgradient(sum(E1,'all'),xp);

%% Scaling

vinf = 6;
sE = std(E1);

%% Loss Vlasov
eq = df1dt + vp.*df1dx - E1.*(df0dv + df1dv);
Loss_Vlasov = mean(eq.^2)./(sE).^2;

%% Loss Poisson

F1 = NN_VlasovPoisson_dec(xp,vp*0+vinf,tp, 1, parameters, PINN.layer, Scaling);
eq = dE1dx + F1;
Loss_Poisson = mean(eq.^2)./sE.^2;
          
%% Initial Condition

[F1,E1] = NN_VlasovPoisson_dec(xi,vi,ti, 1, parameters, PINN.layer, Scaling);

f1 = dlgradient(sum(F1),vi);

sfi = std(f1_ic);
sEi = std(E1_ic);

Zf = ((f1-f1_ic).^2)./sfi.^2;
ZE = ((E1-E1_ic).^2)./sEi.^2;

Loss_IC = mean(Zf) + mean(ZE);

%% Conservation of charge and mass

[F1,E1] = NN_VlasovPoisson_dec(xg,vg,tg, 1, parameters, PINN.layer, Scaling);

E_mean = sum(E1.*wgx,2)./sum(wgx,2);
F_mean = sum(F1.*wgx,2)./sum(wgx,2);

Loss_conservation = mean(E_mean.^2)./sE.^2 + mean(F_mean.^2)./sE.^2;

%% All losses
Losses = [Loss_IC;...
            Loss_Vlasov;...
            Loss_Poisson;...
            Loss_conservation];

Loss = sum(alphas.*Losses);

gradients = dlgradient(Loss,parameters);

end