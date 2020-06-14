%ʵ��KNN�㷨
%%�㷨����
%1����ʼ��ѵ���������
%2��������Լ�������ѵ����������ŷ�Ͼ��룻
%3������ŷ�Ͼ����С��ѵ��������������������
%4��ѡȡŷʽ������С��ǰK��ѵ��������ͳ�����ڸ�����е�Ƶ�ʣ�
%5������Ƶ��������𣬼����Լ��������ڸ����
close all;
clear
clc;

%%�㷨ʵ��
%step1����ʼ��ѵ���������Լ���Kֵ
%����һ����ά���󣬶�ά��ʾͬһ���µĶ�ά����㣬����ά��ʾ���

trainData1=[0 0;0.1 0.3;0.2 0.1;0.2 0.2];%��һ��ѵ������
trainData2=[1 0;1.1 0.3;1.2 0.1;1.2 0.2];%�ڶ���ѵ������
trainData3=[0 1;0.1 1.3;0.2 1.1;0.2 1.2];%������ѵ������
trainData(:,:,1)=trainData1;%���õ�һ���������
trainData(:,:,2)=trainData2;%���õڶ����������
trainData(:,:,3)=trainData3;%���õ������������

trainDim=size(trainData);%��ȡѵ������ά��

testData=[1.6 0.3];%����1�����Ե�

K=7;

%%�ֱ������Լ��и�������ÿ��ѵ�����еĵ��ŷ�Ͼ���
%�Ѳ��Ե���չ�ɾ���
testData_rep=repmat(testData,4,1);
%����������ά�����Ų��Լ�����Ե����չ����Ĳ�ֵƽ��

%diff1=zero(trainDim(1),trianDim(2));
%diff2=zero(trainDim(1),trianDim(2));
%diff3=zero(trainDim(1),trianDim(2));

for i=1:trainDim(3)
    diff1=(trainData(:,:,1)-testData_rep).^2;
    diff2=(trainData(:,:,2)-testData_rep).^2;
    diff3=(trainData(:,:,3)-testData_rep).^2;
end

%��������һά������ŷʽ����
distance1=(diff1(:,1)+diff1(:,2)).^0.5;
distance2=(diff2(:,1)+diff2(:,2)).^0.5;
distance3=(diff3(:,1)+diff3(:,2)).^0.5;

%������һά����ϳ�һ����ά����
temp=[distance1 distance2 distance3];
%�������ά����ת��Ϊһά����
distance=reshape(temp,1,3*4);
%�Ծ����������
distance_sort=sort(distance);
%��һ��ѭ��Ѱ����С��K�����������Ǹ�������ֵ�Ƶ����ߣ������ظ���
num1=0;%��һ����ֵĴ���
num2=0;%�ڶ�����ֵĴ���
num3=0;%��������ֵĴ���
sum=0;%sum1,sum2,sum3�ĺ�
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

fprintf('���Ե㣨%f %f�����ڵ�%d��',testData(1),testData(2),classname);

%%ʹ�û�ͼ��ѵ������Ͳ��Լ���滭����
figure(1);
hold on;
for i=1:4
    plot(trainData1(i,1),trainData1(i,2),'*');
    plot(trainData2(i,1),trainData2(i,2),'o');
    plot(trainData3(i,1),trainData3(i,2),'>');
end
plot(testData(1),testData(2),'x');
text(0.1,0.1,'��һ��');
text(1.1,0.1,'�ڶ���');
text(0.1,1,'������');
