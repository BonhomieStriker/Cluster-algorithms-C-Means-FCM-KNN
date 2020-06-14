%实现KNN算法
%%算法描述
%1、初始化训练集和类别；
%2、计算测试集样本与训练集样本的欧氏距离；
%3、根据欧氏距离大小对训练集样本进行升序排序；
%4、选取欧式距离最小的前K个训练样本，统计其在各类别中的频率；
%5、返回频率最大的类别，即测试集样本属于该类别。
close all;
clear
clc;

%%算法实现
%step1、初始化训练集、测试集、K值
%创建一个三维矩阵，二维表示同一类下的二维坐标点，第三维表示类别

trainData1=[0 0;0.1 0.3;0.2 0.1;0.2 0.2];%第一类训练数据
trainData2=[1 0;1.1 0.3;1.2 0.1;1.2 0.2];%第二类训练数据
trainData3=[0 1;0.1 1.3;0.2 1.1;0.2 1.2];%第三类训练数据
trainData(:,:,1)=trainData1;%设置第一类测试数据
trainData(:,:,2)=trainData2;%设置第二类测试数据
trainData(:,:,3)=trainData3;%设置第三类测试数据

trainDim=size(trainData);%获取训练集的维数

testData=[1.6 0.3];%设置1个测试点

K=7;

%%分别计算测试集中各个点与每个训练集中的点的欧氏距离
%把测试点扩展成矩阵
testData_rep=repmat(testData,4,1);
%设置三个二维矩阵存放测试集与测试点的扩展矩阵的差值平方

%diff1=zero(trainDim(1),trianDim(2));
%diff2=zero(trainDim(1),trianDim(2));
%diff3=zero(trainDim(1),trianDim(2));

for i=1:trainDim(3)
    diff1=(trainData(:,:,1)-testData_rep).^2;
    diff2=(trainData(:,:,2)-testData_rep).^2;
    diff3=(trainData(:,:,3)-testData_rep).^2;
end

%设置三个一维数组存放欧式距离
distance1=(diff1(:,1)+diff1(:,2)).^0.5;
distance2=(diff2(:,1)+diff2(:,2)).^0.5;
distance3=(diff3(:,1)+diff3(:,2)).^0.5;

%将三个一维数组合成一个二维矩阵
temp=[distance1 distance2 distance3];
%将这个二维矩阵转换为一维数组
distance=reshape(temp,1,3*4);
%对距离进行排序
distance_sort=sort(distance);
%用一个循环寻找最小的K个距离里面那个类里出现的频率最高，并返回该类
num1=0;%第一类出现的次数
num2=0;%第二类出现的次数
num3=0;%第三类出现的次数
sum=0;%sum1,sum2,sum3的和
for i=1:K
    for j=1:4
        if distance1(j)==distance_sort(i)
            num1=num1+1;
        end
        if distance2(j)==distance_sort(i)
            num2=num2+1;
        end
        if distance3(j)==distance_sort(i)
            num3=num3+1;
        end
    end
    sum=num1+num2+num3;
    if sum>=K
        break;
    end
end

class=[num1 num2 num3];

classname=find(class(1,:)==max(class));

fprintf('测试点（%f %f）属于第%d类',testData(1),testData(2),classname);

%%使用绘图将训练集点和测试集点绘画出来
figure(1);
hold on;
for i=1:4
    plot(trainData1(i,1),trainData1(i,2),'*');
    plot(trainData2(i,1),trainData2(i,2),'o');
    plot(trainData3(i,1),trainData3(i,2),'>');
end
plot(testData(1),testData(2),'x');
text(0.1,0.1,'第一类');
text(1.1,0.1,'第二类');
text(0.1,1,'第三类');
