
% K = sparse(rows + 1, cols + 1, values);
% spy(K);
function Vv = nullspace(name)
% load data/eigs/squishy_ball_lowlow.mat
load data/eigs/boatv8.mat
% load data/eigs/tmp.mat
% load output/bug/A1.mat;
% na1 = null(A1');

% tilde_K = na1' * K * na1;
% tilde_M = na1' * M * na1;
tilde_K = K;
tilde_M = M;

[Vv, D] = eigs(tilde_K, tilde_M, 100, 'smallestabs');

% diagVMV = diag(Vv' * tilde_M * Vv);
% Vv = Vv ./ sqrt(diagVMV');
% V = na1 * Vv;
% save data/eigs/Q_squishy_ball_lowlow.mat Vv D
save data/eigs/Q_boatv8.mat Vv D

end