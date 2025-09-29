
function Vv = matlab_eigs(name)
load(['data/eigs/' name '.mat']);
tilde_K = K;
% tilde_M = M;
[Vv, D] = eigs(tilde_K, 20, 'smallestabs');
save(['data/eigs/Q_' name '.mat'], 'Vv', 'D');
end