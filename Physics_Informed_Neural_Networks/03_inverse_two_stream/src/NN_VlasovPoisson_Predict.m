
function [f,E,F] = NN_VlasovPoisson_Predict(x,v,t,parameters,Layer,Scaling) 
% NN_VLASOVPOISSON_PREDICT  Evaluate a trained network on a set of points.
%
%   Returns the distribution f, obtained by differentiating the cumulative
%   output F with respect to v, together with E and F, gathered from the GPU
%   and converted to double.


    [F,E] = NN_VlasovPoisson_dec(x,v,t,1,parameters,Layer,Scaling);
    
    f = dlgradient(sum(F,'all'),v);
    
    E = double(extractdata(gather(E)));
    F = double(extractdata(gather(F)));
    f = double(extractdata(gather(f)));

end