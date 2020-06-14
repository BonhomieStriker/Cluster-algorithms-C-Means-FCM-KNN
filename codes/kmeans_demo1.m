clear; close all; clc;

%% 读取数据
% load('k-means_data.mat');
% data= R_rand;
load('data_2.mat');
data = data;

%% 初始化
% 从样本中随机选择K个样本作为初始聚类中心
K = 5;
data_len = size(data, 1);
rand_index = randperm(data_len, K)';
center = data(rand_index, :);
% 指定初始聚类中心
% center = [7.4792 13.8336;
%           9.5291 2.1753;
%           3.8149 8.8972;
%           9.3065 8.6529;
%           11.9423 0.2698];
% center = [11.5208 -0.3655;
%           0.2619 8.6614;
%           2.5069 0.5628;
%           1.4412 7.2833;
%           3.0952 1.6997];


%% 迭代
ite = 0; % 迭代次数
max_ite = 100; % 最大迭代次数
while 1
    % 计算样本与聚类中心距离
    distance = euclidean_distance(data, center);
    % 将距离排序，寻找样本所属类别
    [~, index] = sort(distance, 2, 'ascend');
    
    % 计算新的聚类中心
    center_new = zeros(K, size(data,2));
    for i = 1:K
        data_for_this_class = data(index(:,1)==i, :);
        center_new(i,:) = mean(data_for_this_class, 1);
    end
    
    % 判断是否结束
    ite = ite + 1;
    fprintf('迭代次数：%d\n', ite);
    if(isequal(center_new, center) || (ite>max_ite))
        fprintf('end\n');
        break;
    end
    
    % 更新聚类中心
    center = center_new;
end

%% 绘出结果
figure;
hold on;
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k'];
for i = 1:K
    data_for_this_class = data(index(:,1)==i, :);
    color_for_this_class = [rand(), rand(), rand()];
    h_data = scatter(data_for_this_class(:,1), data_for_this_class(:,2),20,colors(i));
    h_center = scatter(center(i,1), center(i,2), 30, colors(i), 'filled');
end

%% 距离函数
function distance = euclidean_distance(data, center)
    data_len = size(data, 1);
    center_len = size(center, 1);
    distance = zeros(data_len, center_len);
    for i = 1:center_len
        diff = data - repmat(center(i,:), data_len, 1);
        distance(:,i) = sqrt(sum(diff.^2,2));
    end
end