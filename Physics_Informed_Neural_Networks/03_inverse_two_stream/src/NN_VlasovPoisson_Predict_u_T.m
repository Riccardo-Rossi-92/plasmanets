
function [f,E,F,Fv,FT] = NN_VlasovPoisson_Predict_u_T(x,v,t,parameters,Layer,Scaling) 
% NN_VLASOVPOISSON_PREDICT_U_T  Evaluate a trained network on a set of points.
%
%   As NN_VlasovPoisson_Predict, and returns in addition the two cumulative
%   moments Fv and FT produced by the second and third output channels.


    [K,E] = NN_VlasovPoisson_dec_u_T(x,v,t,1,parameters,Layer,Scaling);
    
    F = K(1,:);

    f = dlgradient(sum(F,'all'),v);
    
    E = double(extractdata(gather(E)));
    F = double(extractdata(gather(F)));
    Fv = double(extractdata(gather(K(2,:))));
    FT = double(extractdata(gather(K(3,:))));
    f = double(extractdata(gather(f)));

end