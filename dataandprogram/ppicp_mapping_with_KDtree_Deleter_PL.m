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
title('PPICP with KD-tree, Deleter and P-L点云地图与定位轨迹');

figure; % 绘制定位轨迹俯视图
plot(x,y,'r*-','LineWidth',1.2);
xlabel('x');
ylabel('y');
title('PPICP with KD-tree, Deleter and P-L定位轨迹俯视图');
grid on;

figure; % 绘制MSE迭代曲线图
for i=1:9
    subplot(3,3,i);
    plot(MSE_cell{i},'LineWidth',1.2)
    xlabel('迭代次数i');
    ylabel('MSE');
    title(['第',num2str(i),'次迭代-MSE曲线']);
    axis([0 30 0 3])
    hold on
end

function [Nearestkpoints,min_index] = pointMatch_knn(p,pp,k)
% 从点集P中找出距离点集P'最近的k个点
% 输入参数：
% p: 点集合P
% pp: 点集合P'
% k：寻找距离最近的k个点
% 返回值：
% p_match：数据点匹配后的点集合P
% dist：最近点距离

% 利用Kd-tree，寻找最近的k个点
kd_tree = KDTreeSearcher(p,'BucketSize',10);
[min_index, ~] = knnsearch(kd_tree, pp,'K',k);

% 构建最近点集
temp=p(min_index,:);
Nearestkpoints = reshape(temp,[size(temp,1)/k,k,3]);
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

function normal_vector = calNormalVector(u,R)
% 计算点集U的法向量
% 输入参数：
% u：点集U
% R：用于计算法向量的半径R
% 返回值：
% normal_vector：点集U的法向量（顺序与U对应）

Nearpoints = pointMatch_knn(u,u,80); % 选择最近的80个点
for i = 1:size(u,1) % 遍历点集U中的所有点
    points = reshape(Nearpoints(i,:,:),[size(Nearpoints,2),3]); % 找到第i个点的附近80个点
    dis_vector = points-u(i,:); % 计算差值向量
    dis_square_vector = dis_vector.^2;
    distance_square= sum(dis_square_vector,2); % 计算第i个点到附近点的距离平方
    num = sum(distance_square<R^2); % 统计在R范围内的点个数
    [~,index] = sort(distance_square);
    points = points(index,:); % 选择出在R范围内的点
    meanPoint = mean(points); % 计算中心点
    temp_vector = points-meanPoint;
    covar = (temp_vector'*temp_vector)./num; % 计算协方差矩阵
    [eigA,~]=eig(covar); % 求特征值和特征向量
    normal_vector(i,:) = (eigA(:,2))'; % 计算法向量
end

end

function [p_inter,pp_temp] = findInter(p,pp,tol_dist,tol_cos_beta)
% 寻找正交点并剔除离群点
% 输入参数：
% p: 待匹配点集合P
% pp: 待移动点集合P'
% tol_dist：距离判定阈值
% tol_cos_beta：角度判定阈值
% 返回值：
% p_inter：正交点
% pp_temp：删去离群点的点集合P’
if (nargin==2)
    tol_dist = 2;
    tol_cos_beta = 0.1;
end % 参数default值

index = 1; % 初始化
normal_vector_p = calNormalVector(p,0.2); % 计算点集P的法向量
normal_vector_pp = calNormalVector(pp,0.2); % 计算点集P'的法向量
[Nearest2points,min_index] = pointMatch_knn(p,pp,2); % knn最近邻搜索
normal_vector_p = normal_vector_p(min_index,:); % 对法向量重新排序，以保证对应关系
normal_vector_p = reshape(normal_vector_p,[180,size(normal_vector_p,1)/180,3]); % 数组形状重构

for i =1:size(Nearest2points,1) % 遍历每个点
    Line_vector = Nearest2points(i,1,:)-Nearest2points(i,2,:);
    Line_vector = reshape(Line_vector,[1,3]); % 计算直线向量
    h_vector = pp(i,:)-reshape(Nearest2points(i,2,:),[1,3]); % 计算斜向量
    length = norm(Line_vector); % 计算线段长
    dist = cross(Line_vector,h_vector)/length; % 计算点到直线的距离
    
    cos_beta1 = abs(reshape(normal_vector_p(i,1,:),[1,3])*normal_vector_pp(i,:)'); % 计算法向量夹角
    cos_beta2 = abs(reshape(normal_vector_p(i,2,:),[1,3])*normal_vector_pp(i,:)'); % 计算法向量夹角
    
    if dist < tol_dist  & cos_beta1 > tol_cos_beta & cos_beta2 > tol_cos_beta % 保留标准
        rate = (Line_vector*h_vector')/length/length; % 计算正交点在线段中的比例
        p_inter(index,:)=reshape(rate*Nearest2points(i,2,:)+(1-rate)*Nearest2points(i,1,:),[1,3]); % 计算正交点坐标
        pp_temp(index,:)=pp(i,:); % 更新pp，与之对应
        index = index + 1;
    end
end
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

% 根据最近邻域规则建立p和pp的关系，并寻找正交点
[equ_p,pp_temp] = findInter(p,pp,2,0.1);

for i=1:i_max % 最大迭代次数
    
    % SVD分解，估计旋转平移量
    [R,t] = svdSolution(equ_p,pp_temp);
    
    tform_temp = [R,t;0,0,0,1];% 得到刚体变换矩阵
    tform = tform_temp * tform;% 得到累计tform_step
    
    % 对点集合P'的点进行旋转平移
    pp = (R*pp'+t)'; % 对pp作变换
    
    % 重新关联，寻找正交点
    [equ_p,pp_temp] = findInter(p,pp,2,0.1);
    
    %计算均方误差MSE
    MSE = calMSE(equ_p,pp_temp);
    MSEplt(i) = MSE; % 绘图用
    
    if MSE<tol_MSE
        break % 均方误差小于阈值，结束迭代
    end
end

end
