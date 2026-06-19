
function [f,E,F,Fv,FT] = NN_VlasovPoisson_Predict_u_T(x,v,t,parameters,Layer,Scaling) 

    [K,E] = NN_VlasovPoisson_dec_u_T(x,v,t,1,parameters,Layer,Scaling);
    
    F = K(1,:);

    f = dlgradient(sum(F,'all'),v);
    
    E = double(extractdata(gather(E)));
    F = double(extractdata(gather(F)));
    Fv = double(extractdata(gather(K(2,:))));
    FT = double(extractdata(gather(K(3,:))));
    f = double(extractdata(gather(f)));

end