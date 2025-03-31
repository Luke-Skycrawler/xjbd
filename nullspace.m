
% K = sparse(rows + 1, cols + 1, values);
% spy(K);

load data/eigs/bug.tobj.mat
% load output/bug/A1.mat;
% na1 = null(A1');

% tilde_K = na1' * K * na1;
% tilde_M = na1' * M * na1;
tilde_K = K;
tilde_M = M;

[Vv, D] = eigs(tilde_K, tilde_M, 20, 'smallestabs');

% diagVMV = diag(Vv' * tilde_M * Vv);
% Vv = Vv ./ sqrt(diagVMV');
% V = na1 * Vv;
save data/eigs/Q_bug.tobj.mat V D

