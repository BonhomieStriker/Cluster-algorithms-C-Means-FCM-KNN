function y = myKNN(X, X_train, y_train, K)
%KNN k-Nearest Neighbors Algorithm.
%
%   INPUT:  X:         testing sample features, P-by-N_test matrix.
%           X_train:   training sample features, P-by-N matrix.
%           y_train:   training sample labels, 1-by-N row vector.
%           K:         the k in k-Nearest Neighbors
%
%   OUTPUT: y    : predicted labels, 1-by-N_test row vector.

[~,N_test] = size(X);

predicted_label = zeros(1,N_test);
for i=1:N_test
    [dists, neighbors] = top_K_neighbors(X_train,y_train,X(:,i),K); 
    % calculate the K nearest neighbors and the distances.
    predicted_label(i) = recog(y_train(neighbors),max(y_train));
    % recognize the label of the test vector.
end

y = predicted_label;

