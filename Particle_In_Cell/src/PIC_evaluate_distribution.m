
function f = PIC_evaluate_distribution(x,v,Geo)
% PIC_EVALUATE_DISTRIBUTION  Histogram the particles onto the (x,v) grid.
%
%   Returns f on the Nv-by-Nx diagnostic grid, normalised so that the
%   integral of f over v is the density.  Diagnostics only: this grid plays
%   no part in the dynamics.


    xg = Geo.xn;
    vg = Geo.vn;
    
    Nx = length(xg);
    Nv = length(vg);
    
    x_edges = linspace(Geo.range_xn(1), Geo.range_xn(2), Nx+1);
    v_edges = linspace(Geo.range_vn(1), Geo.range_vn(2), Nv+1);
    
    f = histcounts2(x, v, x_edges, v_edges,"Normalization","pdf").*Geo.Ln;
    f = f';

end








