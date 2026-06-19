
function [f,E,F] = NN_VlasovPoisson_Predict(x,v,t,parameters,Layer,Scaling) 

    [F,E] = NN_VlasovPoisson_dec(x,v,t,1,parameters,Layer,Scaling);
    
    f = dlgradient(sum(F,'all'),v);
    
    E = double(extractdata(gather(E)));
    F = double(extractdata(gather(F)));
    f = double(extractdata(gather(f)));

end