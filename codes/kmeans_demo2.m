clear; close all; clc;
% ������Ŀk
K = 5;

load('k-means_data.mat');
data= R_rand; % ֱ�Ӵ洢��data������

x = data(:,1);
y = data(:,2);

% �������ݣ�2άɢ��ͼ
% x,y: Ҫ���Ƶ����ݵ�  20:ɢ���С��ͬ����Ϊ20  'blue':ɢ����ɫΪ��ɫ
s = scatter(x, y, 20, 'blue');
title('ԭʼ���ݣ���Ȧ����ʼ���ģ����');

% ��ʼ������
sample_num = size(data, 1);       % ��������
sample_dimension = size(data, 2); % ÿ����������ά��

% �����ֶ�ָ�����ĳ�ʼλ��
% clusters = zeros(K, sample_dimension);
% clusters(1,:) = [-3,1];
% clusters(2,:) = [2,4];
% clusters(3,:) = [-1,-0.5];
% clusters(4,:) = [2,-3];
% ���ĸ���ֵ�������������ݵľ�ֵ������һЩС��������ӵ���ֵ��
clusters = zeros(K, sample_dimension);
minVal = min(data); % ��ά�ȼ�����Сֵ
maxVal = max(data); % ��ά�ȼ������ֵ
for i=1:K
    clusters(i, :) = minVal + (maxVal - minVal) * rand();
end 


hold on; % ���ϴλ�ͼ��ɢ��ͼ�������ϣ�׼���´λ�ͼ
% ���Ƴ�ʼ����
scatter(clusters(:,1), clusters(:,2), 'red', 'filled'); % ʵ��Բ�㣬��ʾ���ĳ�ʼλ��

c = zeros(sample_num, 1); % ÿ�����������صı��

PRECISION = 0.1;


iter = 3000; % �ٶ�������3000��
% Stochastic Gradient Descendant ����ݶ��½���SGD����K-means��Ҳ����Competitive Learning�汾
basic_eta = 1;  % learning rate
% ��ʼ��
acc_err = 0;  % �ۼ����
for i=1:iter
    pre_acc_err = acc_err;  % ��һ�ε����У��ۼ����
    acc_err = 0;  % �ۼ����
    for j=1:sample_num
        x_j = data(j, :);     % ȡ�õ�j���������ݣ�����������stochastic����

        % ���д��ĺ�x������룬�ҵ������һ�����Ƚϴ��ĵ�x��ģ����
        gg = repmat(x_j, K, 1);
        gg = gg - clusters;
        tt = arrayfun(@(n) norm(gg(n,:)), (1:K)');
        [minVal, minIdx] = min(tt);

        % ���´��ģ�������Ĵ���(winner)������x������ etaΪѧϰ��.
        eta = basic_eta/i;
        delta = eta*(x_j-clusters(minIdx,:));
        clusters(minIdx,:) = clusters(minIdx,:) + delta;
        acc_err = acc_err + norm(delta);
        c(j)=minIdx;
    end
    
    disp(['��', num2str(i), '�ε����ۼ���', num2str(abs(acc_err-pre_acc_err))]);
    
    if(rem(i,100) ~= 0)
        continue
    end
    figure;
    f = scatter(x, y, 20, 'blue');
    hold on;
    scatter(clusters(:,1), clusters(:,2), 'filled'); % ʵ��Բ�㣬��ʾ���ĳ�ʼλ��
    title(['��', num2str(i), '�ε���']);
    if (abs(acc_err-pre_acc_err) < PRECISION)
        disp(['�����ڵ�', num2str(i), '�ε���']);
        break;
    end
    pre_acc_err = acc_err;
end


disp('done');