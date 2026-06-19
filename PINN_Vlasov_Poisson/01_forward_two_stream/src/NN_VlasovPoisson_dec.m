function [Xk, Xm, parameters] = NN_VlasovPoisson_dec(x,v,t,Predict,parameters,Layer,Scaling)


if Predict == 0
    
    %% Scaling
    v = (v-Scaling.v(1,1))./Scaling.v(2,1);
    t = (t-Scaling.t(1,1))./Scaling.t(2,1);

    %% Input
    Xk = [cos(x); sin(x); v; t];
    Xm = [cos(x); sin(x); t];
    g = v+1;

    %% Hidden Layers 

    for i = 1 : length(Layer)

        fan_in  = size(Xk, 1);
        fan_out = Layer(i);
        
        % initialisation hidden layers
        parameters.("k"+i).weights = dlarray(randn([fan_out, fan_in]) * ...
            sqrt(2 / (fan_in + fan_out)));
        parameters.("k"+i).bias = dlarray(zeros([Layer(i) 1]));

        % fully-conneted activation function
        Xk = fullyconnect(Xk,parameters.("k"+i).weights,parameters.("k"+i).bias);

        % SiLU (swish) activation function
        Xk = tanh(Xk);

    end

    for i = 1 : length(Layer)
        
        fan_in  = size(Xm, 1);
        fan_out = Layer(i);

        % initialisation hidden layers
        parameters.("m"+i).weights = dlarray(randn([fan_out, fan_in]) * ...
            sqrt(2 / (fan_in + fan_out)));
        parameters.("m"+i).bias = dlarray(zeros([Layer(i) 1]));

        % fully-conneted activation function
        Xm = fullyconnect(Xm,parameters.("m"+i).weights,parameters.("m"+i).bias);

        % SiLU (swish) activation function
        Xm = tanh(Xm);
    end

    %% Output Layer

    % Output initialisation
    parameters.outk.weights = dlarray(randn([1 size(Xk,1)]));
    parameters.outk.bias = dlarray(zeros([1 1]));

    parameters.outm.weights = dlarray(randn([1 size(Xm,1)]));
    parameters.outm.bias = dlarray(zeros([1 1]));

    parameters.scaling.F = dlarray(Scaling.f0);
    parameters.scaling.E = dlarray(Scaling.E0);

    % Output fully connected
    Xk = fullyconnect(Xk,parameters.outk.weights,parameters.outk.bias);
    Xm = fullyconnect(Xm,parameters.outm.weights,parameters.outm.bias);

    % Non negativity
    Xk = log(1+exp(Xk)).*g.*parameters.scaling.F;
    Xm = Xm.*parameters.scaling.E;

else

    %% Scaling
    v = (v-Scaling.v(1,1))./Scaling.v(2,1);
    t = (t-Scaling.t(1,1))./Scaling.t(2,1);

    %% Input
    Xk = [cos(x); sin(x); v; t];
    Xm = [cos(x); sin(x); t];
    g = v+1;
    
    %% Hidden Layers 

    for i = 1 : length(Layer)
        % fully-conneted activation function
        Xk = fullyconnect(Xk,parameters.("k"+i).weights,parameters.("k"+i).bias);
        Xk = tanh(Xk);
    end

    for i = 1 : length(Layer)
        % fully-conneted activation function
        Xm = fullyconnect(Xm,parameters.("m"+i).weights,parameters.("m"+i).bias);
        Xm = tanh(Xm);
    end

    %% Output Layer

    % Output fully connected
    Xk = fullyconnect(Xk,parameters.outk.weights,parameters.outk.bias);
    Xm = fullyconnect(Xm,parameters.outm.weights,parameters.outm.bias);

    % Non negativity
    Xk = log(1+exp(Xk)).*g.*parameters.scaling.F;
    Xm = Xm.*parameters.scaling.E;

end
end