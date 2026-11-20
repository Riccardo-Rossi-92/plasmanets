function PIC = PIC_geometry_1D(PIC)
% PIC_GEOMETRY_1D  Build the 1D1V grid from the configuration fields.
%
%   Fills PIC.geo with the domain length, the cell sizes dx and dv, the node
%   vectors xn and vn and the meshgrid (xgn,vgn).  The velocity grid is used
%   for the initial condition and for diagnostics only, never for the
%   dynamics.


Geo = PIC.geo;

Geo.Nx = PIC.config.Nx; Geo.Nv = PIC.config.Nv;

Geo.Ln = Geo.range_xn(2)-Geo.range_xn(1);
Geo.dx = Geo.Ln./(Geo.Nx);
Geo.dv = (Geo.range_vn(2)-Geo.range_vn(1))./Geo.Nv;

Geo.xn = linspace(Geo.range_xn(1),Geo.range_xn(2),Geo.Nx);
Geo.vn = linspace(Geo.range_vn(1),Geo.range_vn(2),Geo.Nv);

[Geo.xgn,Geo.vgn] = meshgrid(Geo.xn,Geo.vn);

PIC.geo = Geo;

end