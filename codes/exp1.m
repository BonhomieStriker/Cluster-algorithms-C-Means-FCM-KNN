clear
close all
clc

%% Dimensionality reduction using PCA
load('data_2.mat');
% Improvement process
% for n = 1:size(data,1)
%     temp = reshape(data(n,:), [28 28]);
%     temp = temp';
%     for j = 1:7
%         for i = 1:7
%             img(i,j) = sum(sum(temp(i:i+3,j:j+3)));
%         end
%     end
%     data_acc(n, :) = reshape(img, 1, []);
% end
% [coeff,score,latent] = pca(data_acc);

[coeff,score,latent] = pca(data);
ratio = cumsum(latent)/sum(latent); 
p = 3; % choose the top p features with greatest eigenvalues in cov(data)
data_pca = score(:,1:p);
K = 7; % number of cluster centers

% imshow(reshape(data(48200,:), [28 28]));
% for i = 1:size(data, 1)
%     namestring = ['D:\Tsinghua\研究生课程\模式识别\experiments\img\img', num2str(i), '.jpg'];
%     imwrite(reshape(data(i,:), [28 28]), namestring);
% end
% Through observation we find there are 8 clusters:
%0(1-5923,5923), 1(5924-12665,6742), 2(12666 - 18623,5958), 3(18624-24754, 6131)
%4(24755-30596,5842), 5(30597-36017,5421), 6(36018-41935, 5918), 7(41936-48200, 6265)
% True_cluster{1} = data_pca(1:5923,:);
% True_cluster{2} = data_pca(5924:12665,:);
% True_cluster{3} = data_pca(12666:18623,:);
% True_cluster{4} = data_pca(18624:24754,:);
% True_cluster{5} = data_pca(24755:30596,:);
% True_cluster{6} = data_pca(30597:36017,:);
% True_cluster{7} = data_pca(36018:41935,:);
% True_cluster{8} = data_pca(41936:48200,:);
% figure(1)
% for i = 1:8
%     X = True_cluster{i}(:,1);
%     Y = True_cluster{i}(:,2);
%     Z = True_cluster{i}(:,3);
%     scatter3(X, Y, Z);hold on
% end
% title('True Clustering Result')

%% Algorithm 1: C Means Clustering
% Initialize the cluster centers
sample_num = size(data_pca, 1);       % number of samples
sample_dimension = size(data_pca, 2); % number of features

clusters = zeros(K, sample_dimension);
minVal = min(data_pca); 
maxVal = max(data_pca); 
for i=1:K
    clusters(i, :) = minVal + (maxVal - minVal) * rand();
end 

c = zeros(sample_num, 1); % cluster labels for samples
PRECISION = 0.1;


iter = 100; % iteration times
% Stochastic Gradient Descendant C Means
basic_eta = 1;  % learning rate
acc_err = 0;  % initialize the accumulative error
for i=1:iter
    pre_acc_err = acc_err;  % acc_error in last iteration
    acc_err = 0;  % accumulative error
    for n=1:sample_num
        x_j = data_pca(n, :);
        % Calculate the distance between x and all cluster centers, then
        % find the nearest center
        gg = repmat(x_j, K, 1);
        gg = gg - clusters;
        tt = arrayfun(@(n) norm(gg(n,:)), (1:K)');
        [minVal, minIdx] = min(tt);
        % Update the cluster centers: drag the nearest center to x
        eta = basic_eta/i;
        delta = eta*(x_j-clusters(minIdx,:));
        clusters(minIdx,:) = clusters(minIdx,:) + delta;
        acc_err = acc_err + norm(delta);
        c(n)=minIdx;
    end
    
    disp(['Accumulative error in iteration ', num2str(i),...
        ' : ', num2str(abs(acc_err-pre_acc_err))]);
    
    if(rem(i,100) ~= 0)
        continue
    end
    if (abs(acc_err-pre_acc_err) < PRECISION)
        disp(['Converge to', num2str(i), 'th iteration']);
        break;
    end
    pre_acc_err = acc_err;
end
disp('done');

%plot the clustering results
figure(2)
for i = 1:8
    cluster1{i} = data_pca(c == i,:);
    X = cluster1{i}(:,1);
    Y = cluster1{i}(:,2);
    Z = cluster1{i}(:,3);
    scatter3(X, Y, Z);hold on
end
title('Clustering Result of C-Means ')

% Evaluation
for i = 1:8
    center1{i} = mean(cluster1{i}(:,:),1);
    dist1(i) = 0;
    temp = cluster1{i};
    temp = (temp - center1{i}).^2;
    dist1(i) = sum(sum(temp))/size(temp,1);
end
temp = zeros(8,p);
for i = 1:8
    temp(i,:) = center1{i};
end
for i = 1:8
    dij1 = (temp - temp(i,:)).^2;
    dist11(:,i) = sum(dij1,2);
end

%% Algorithm 2: Fuzzy C - Means (FCM)
opt = [2 3000 1e-10 1];
% [centers,U,objFunc] = fcm(data_pca, K, opt);
[center, U, obj_fcn] = myFCM(data_pca, K, opt);

%plot the clustering result
[maxU, label] = max(U, [], 2);
figure(3)
for i = 1:8
    cluster2{i} = data_pca(label == i,:);
    X = cluster2{i}(:,1);
    Y = cluster2{i}(:,2);
    Z = cluster2{i}(:,3);
    scatter3(X, Y, Z);hold on
end
title('Clustering Result of FCM')
% Evaluation
for i = 1:8
    center2{i} = mean(cluster2{i}(:,:),1);
    dist2(i) = 0;
    temp = cluster2{i};
    temp = (temp - center2{i}).^2;
    dist2(i) = sum(sum(temp))/size(temp,1);
end
temp = zeros(8,p);
for i = 1:8
    temp(i,:) = center2{i};
end
for i = 1:8
    dij2 = (temp - temp(i,:)).^2;
    dist22(:,i) = sum(dij2,2);
end
%% Algorithm 3: K - Nearest Neighbor (KNN)
% set the training sets and test sets
trainData(:,:,1)=data_pca(1:2000,:);
trainData(:,:,2)=data_pca(5924:7923,:);
trainData(:,:,3)=data_pca(12666:14665,:);
trainData(:,:,4)=data_pca(18624:20623,:);
trainData(:,:,5)=data_pca(24755:26754,:);
trainData(:,:,6)=data_pca(30597:32596,:);
trainData(:,:,7)=data_pca(36018:38017,:);
trainData(:,:,8)=data_pca(41936:43935,:);
trainDim=size(trainData);

testData = vertcat(data_pca(2001:5923,:), data_pca(7924:12665,:),...
    data_pca(14666:18623,:), data_pca(20624:24754,:),...
    data_pca(26755:30596,:), data_pca(32597:36017,:),...
    data_pca(38018:41935,:), data_pca(43936:48200,:));
Labels = zeros(size(testData,1),1);

k = 5; % nearest k samples
for n = 1:size(testData,1)
    for m = 1:trainDim(3)
        temp = (trainData(:,:, m) - testData(n, :)).^2;
        distTable(:,:,m) = sqrt(sum(temp, 2));
        distTable(:,:,m) = sort(distTable(:,:,m));
    end
    distList = sort(reshape(distTable,[], 1));
    labelList = zeros(K,1);
    for i = 1:k
        for j = 1:trainDim(3)
            if(sum(distList(i)==distTable(:,:,j)))
                % add the exp term to avoid the contradictory case
                labelList(j) = labelList(j) + 1-0.01*exp(i);
            end
        end
    end
    Labels(n) = find(labelList == max(labelList));
end

figure(4)
for i = 1:8
    cluster3{i} = testData(find(Labels == i),:);
    X = cluster3{i}(:,1);
    Y = cluster3{i}(:,2);
    Z = cluster3{i}(:,3);
    scatter3(X, Y, Z);hold on
end
title('Clustering Result of KNN')
        
% Evaluation
for i = 1:8
    center3{i} = mean(cluster3{i}(:,:),1);
    dist3(i) = 0;
    temp = cluster3{i};
    temp = (temp - center3{i}).^2;
    dist3(i) = sum(sum(temp))/size(temp,1);
end
temp = zeros(8,p);
for i = 1:8
    temp(i,:) = center3{i};
end
for i = 1:8
    dij3 = (temp - temp(i,:)).^2;
    dist33(:,i) = sum(dij3,2);
end
