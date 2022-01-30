clear;
clc;
close all;

p_ply = pcread('0.ply');

tform = eye(4,4); % 刚体变换矩阵初始化
robot_tf{1} = tform; %用于储存各个tform
for i = 1:9
    % read
    str = [num2str(i) , '.ply']; % 点云文件名
    pp_ply = pcread(str);  % 读入新点云
    p = p_ply.Location; % 点集合P
    pp = pp_ply.Location; % 点集合P'
   
    % icp
    [tform,pp,MSE_cell{i}]=icp(p,pp,tform,100,0.01);
    robot_tf{i+1} = tform; % 记录当前tform
    
    % merge  
    p_ply = pcmerge(pointCloud(p), pointCloud(pp), 0.001); % 点云合并
end

% x,y,z的坐标即为平移向量t
for i=1:10
    x(i) = robot_tf{i}(1,4);    
    y(i) = robot_tf{i}(2,4);   
    z(i) = robot_tf{i}(3,4); 
    theta = asin(robot_tf{i}(2,1))*180/pi;
    disp(['第',num2str(i),'个点坐标为(',num2str(x(i)),',',num2str(y(i)),',',num2str(z(i)),')，角度为',num2str(theta),'°'])
end


figure; % 绘制点云地图与定位轨迹
pcshow(p_ply, 'MarkerSize', 20);
hold on;
plot3(x,y,z,'r*-','LineWidth',1.2);
title('PPICP with KD-tree点云地图与定位轨迹');

figure; % 绘制定位轨迹俯视图
plot(x,y,'r*-','LineWidth',1.2);
xlabel('x');
ylabel('y');
title('PPICP with KD-tree定位轨迹俯视图');
grid on;

figure; % 绘制MSE迭代曲线图
for i=1:9
    subplot(3,3,i);
    plot(MSE_cell{i},'LineWidth',1.2)
    xlabel('迭代次数i');
    ylabel('MSE');
    title(['第',num2str(i),'次迭代-MSE曲线']);
    axis([0 30 0 0.9]) 
    hold on
end


function p_match = pointMatch_knn(p,pp) 
% 根据最近邻域规则建立p和pp的关系（knn）
% 输入参数：
% p: 点集合P
% pp: 点集合P'
% 返回值：
% p_match：数据点匹配后的点集合P

% 利用Kd-tree，寻找最近点
kd_tree = KDTreeSearcher(p,'BucketSize',10);
[min_index, dist] = knnsearch(kd_tree, pp);
p_match = p(min_index,:); % p和pp最近点对应
end

function p_match = pointMatch_dist(p,pp) 
% 根据最近邻域规则建立p和pp的关系（欧氏距离）
% 输入参数：
% p: 点集合P
% pp: 点集合P'
% 返回值：
% p_match：数据点匹配后的点集合P

% 寻找最近点
p_match = zeros(180,3);% 关系矩阵初始化
for i=1:size(pp,1)
    dis_vector = p-pp(i,:); % 差值向量
    dis_square_vector = dis_vector.^2;
    distance_square= sum(dis_square_vector,2); % 计算P中各点到P'中第i点的距离平方
    [~,min_index]=min(distance_square); % 找到距离最近的点
    p_match(i,:) = p(min_index,:); % p和pp最近点对应
end
end

function [R,t] = svdSolution(p,pp)
% 线性代数法（SVD分解）
% 输入参数：
% p: 待匹配点集合P
% pp: 待移动点集合P'（P和P'已最近邻点关联，一一对应）
% 返回值：
% R：旋转矩阵
% t：平移向量

% 计算质心位置
pp_center = mean(pp);
p_center = mean(p);

% 计算每个点的去质心坐标
q = p-p_center;
qq = pp-pp_center;

% 计算矩阵W
W=zeros(3,3);
for i = 1:size(pp,1)
    W = W+qq(i,:)'*q(i,:); % 求矩阵W
end

[U,~,V] = svd(W); % svd分解

R = V*U';% 求出旋转矩阵R
t = p_center'- R*pp_center';% 求出平移矩阵

end

function MSE = calMSE(p,pp)
% 计算均方误差MSE
% 输入参数：
% p: 点集合P
% pp: 点集合P'
% 返回值：
% MSE：两个点集合的均方误差

delta_p = p-pp;
delta_p_square = sum(delta_p.^2,2);
MSE = mean(delta_p_square);
end

function [tform, pp, MSEplt]=icp(p, pp, tform, i_max, tol_MSE)
% icp算法
% 输入参数：
% p: 待匹配点集合P
% pp: 待移动点集合P'
% tform: 粗配准刚体变换矩阵
% i_max: 最大迭代次数
% tol_MSE: 均方误差阈值
% 返回值：
% tform：迭代后的精配准刚体变换矩阵
% pp：变换后的点集合P'
% MSEplt：MSE的各次迭代值记录，用于绘图

% 进行初步变换，粗配准
R = tform(1:3,1:3); % 得到旋转矩阵
t = tform(1:3,4); % 得到平移向量
pp = (R*pp'+t)'; % 对点集P'进行变换

% 根据最近邻域规则建立p和pp的关系
p_match = pointMatch_knn(p,pp); % KD-tree加速搜索
% p_match = pointMatch_dist(p,pp); % 欧式距离搜索

for i=1:i_max % 最大迭代次数
    % SVD分解，估计旋转平移量
    [R,t] = svdSolution(p_match,pp);
    tform_temp = [R,t;0,0,0,1];% 得到刚体变换矩阵
    tform = tform_temp * tform;% 得到累计tform_step
    
    % 对点集合P'的点进行旋转平移
    pp = (R*pp'+t)'; % 对pp作变换

    % 重新关联
    p_match = pointMatch_knn(p,pp); % KD-Tree加速搜索
    % p_match = pointMatch_dist(p,pp); % 欧式距离搜索
    
    %计算均方误差MSE
    MSE = calMSE(p_match,pp);
    MSEplt(i) = MSE; % 绘图用
    
    if MSE<tol_MSE
        break % 均方误差小于阈值，结束迭代
    end
end

end 



