function [Xk, Xm, parameters] = NN_VlasovPoisson_dec_u_T(x,v,t,Predict,parameters,Layer,Scaling)
% NN_VLASOVPOISSON_DEC_U_T  Two-branch network, three cumulative outputs.
%
%   As NN_VlasovPoisson_dec, but branch k returns three channels: the
%   cumulative distribution F and its first two velocity moments, which are
%   compared with the measured momentum and energy in the enriched inverse
%   problem.  Branch m returns the electric field E(x,t).  The softplus is
%   applied to the first and third channels only; the second may change sign.
%
%   Predict = 0 initialises the weights and returns them in parameters;
%   Predict = 1 evaluates the network with the weights given.



if Predict == 0
    
    %% Scaling
    % x = (x-Scaling.x(1,1))./Scaling.x(2,1);
    v = (v-Scaling.v(1,1))./Scaling.v(2,1);
    t = (t-Scaling.t(1,1))./Scaling.t(2,1);

    %% Input
   
    Xk = [cos(x); sin(x); v; t];
    Xm = [cos(x); sin(x); t];

    %% Hidden Layers 

    for i = 1 : length(Layer)

        % initialisation hidden layers
        parameters.("k"+i).weights = dlarray(randn([Layer(i) size(Xk,1)])*sqrt(1/Layer(i)));
        parameters.("k"+i).bias = dlarray(zeros([Layer(i) 1]));

        % fully connected layer
        Xk = fullyconnect(Xk,parameters.("k"+i).weights,parameters.("k"+i).bias);

        % tanh activation function
        Xk = tanh(Xk);

    end

    for i = 1 : length(Layer)

        % initialisation hidden layers
        parameters.("m"+i).weights = dlarray(randn([Layer(i) size(Xm,1)])*sqrt(1/Layer(i)));
        parameters.("m"+i).bias = dlarray(zeros([Layer(i) 1]));

        % fully connected layer
        Xm = fullyconnect(Xm,parameters.("m"+i).weights,parameters.("m"+i).bias);

        % tanh activation function
        Xm = tanh(Xm);
    end

    %% Output Layer

    % Output initialisation
    parameters.outk.weights = dlarray(randn([3 size(Xk,1)]));
    parameters.outk.bias = dlarray(zeros([3 1]));

    parameters.outm.weights = dlarray(randn([1 size(Xm,1)]));
    parameters.outm.bias = dlarray(zeros([1 1]));

    % Output fully connected
    Xk = fullyconnect(Xk,parameters.outk.weights,parameters.outk.bias);
    Xm = fullyconnect(Xm,parameters.outm.weights,parameters.outm.bias);

    % Non negativity
    Xk = log(1+exp(Xk));

    parameters.scaling.F = dlarray(Scaling.f0);
    parameters.scaling.E = dlarray(Scaling.E0);

else

    %% Scaling
    % x = (x-Scaling.x(1,1))./Scaling.x(2,1);
    v = (v-Scaling.v(1,1))./Scaling.v(2,1);
    t = (t-Scaling.t(1,1))./Scaling.t(2,1);

    %% Input
    Xk = [cos(x); sin(x); v; t];
    Xm = [cos(x); sin(x); t];
    g = v+1;
    
    %% Hidden Layers 

    for i = 1 : length(Layer)
        % fully connected layer
        Xk = fullyconnect(Xk,parameters.("k"+i).weights,parameters.("k"+i).bias);
        % tanh activation function
        Xk = tanh(Xk);
    end

    for i = 1 : length(Layer)
        % fully connected layer
        Xm = fullyconnect(Xm,parameters.("m"+i).weights,parameters.("m"+i).bias);
        % tanh activation function
        Xm = tanh(Xm);
    end

    %% Output Layer

    % Output fully connected
    Xk = fullyconnect(Xk,parameters.outk.weights,parameters.outk.bias);
    Xm = fullyconnect(Xm,parameters.outm.weights,parameters.outm.bias);

    % Non negativity
    Xk = log(1+exp(Xk)).*g.*parameters.scaling.F.*[1;0;1] + ...
        Xk.*g.*parameters.scaling.F.*[0;1;0];
    Xm = Xm.*parameters.scaling.E;

end
end