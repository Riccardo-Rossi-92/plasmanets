
function [x,v,PIC] = PIC_generate_particles(PIC)
% PIC_GENERATE_PARTICLES  Load particles cell by cell, uniformly at random.
%
%   Places round(f*dx*dv/Ln*N) particles uniformly inside each phase-space
%   cell.  Simple but noisy; main_pic.m uses sample_particles_from_f instead,
%   which inverts the CDF at Hammersley quantiles.  Kept for reference.


Nxv = (PIC.IC.f.*PIC.geo.dv.*PIC.geo.dx./PIC.geo.Ln).*PIC.config.N;
Nxv = round(Nxv);

PIC.config.N = sum(round(Nxv),'all');

x = []; v = [];

x0 = PIC.geo.range_xn(1); dx = PIC.geo.dx;

for ix = 1 : size(PIC.IC.f,2)

    v0 = PIC.geo.range_vn(1); dv = PIC.geo.dv;

    for iv = 1 : size(PIC.IC.f,1)

        xt = rand(Nxv(iv,ix),1).*dx + x0;
        vt = rand(Nxv(iv,ix),1).*dv + v0;

        x = [x; xt];
        v = [v; vt];

        v0 = v0 + dv;

    end

    x0 = x0 + dx;
end

end

