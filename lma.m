function Vv = lma(name)
load(['data/eigs_lma/' name '.mat']);
tilde_K = K;
tilde_M = M;

[Vv, D] = eigs(tilde_K, tilde_M, 30, 'smallestabs');
save(['data/eigs_lma/Q_' name '.mat'], 'Vv', 'D');

end