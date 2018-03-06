%start code for project #1: linear regression
%pattern recognition, CSE583/EE552
%Weina Ge, Aug 2008
%Christopher Funk, Jan 2018
clc
clear
close all

addpath export_fig/

%load the data points
load data.mat
t = t';
%plot the groud truth curve
figure(1)
clf
hold on;
xx = linspace(1,4*pi,50);
yy = sin(.5*xx);
err = ones(size(xx))*0.3;
% plot the x and y color the area around the line by err (here the std)
h = shadedErrorBar(x,y,err,{'b-','color','r','LineWidth',2},0);
%plot the noisy observations
plot(x,t,'bo','MarkerSize',8,'LineWidth',1.5);
hold off; 

% Make it look good
grid on;
set(gca,'FontWeight','bold','LineWidth',2)
xlabel('x')
ylabel('t')

% Save the image into a decent resolution
% export_fig sampleplot -png -transparent -r150

%% Start your curve fitting program here
% Start with determining the coefficient of the polynomial 
order = 40; % the order of the polynomial
N = length(x);
range = max(x) - min(x);

% energy minimization without regulation
M = order;
X = zeros(N, M+1);
% compute the noisy data from 0-th to the M-th order
for i = 1:M+1
    X(:,i) = x.^(i-1);
end
w = (X'*X)\X'*t;

aa = linspace(min(x), max(x), N)';
AA = zeros(N, M+1);
for i = 1:M+1
    AA(:,i) = aa.^(i-1);
end

Yn1 = AA*w;
err1 = 0.5*(X*w-t)'*(X*w-t);

figure(2)
hold on
shadedErrorBar(x,y,err,{'b-','color','r','LineWidth',2},0);
plot(aa, Yn1, 'g', 'LineWidth',2);
plot(x,t,'bo','MarkerSize',8,'LineWidth',1.5);
hold off

grid on;
set(gca,'FontWeight','bold','LineWidth',2)
xlabel('x')
ylabel('t')
% export_fig p1_order40 -png -transparent -r150

% energy minimization with regularization
lamda = exp(-10);

w_r = (lamda * eye(M+1)+ X'*X) \ X' * t;
err2 = 0.5*(X*w_r-t)'*(X*w_r-t)+lamda/2*(w_r'*w_r);
Yn2 = AA * w_r;
figure(3)
hold on
shadedErrorBar(x,y,err,{'b-','color','r','LineWidth',2},0);
plot(aa, Yn2, 'g', 'LineWidth',2);
plot(x,t,'bo','MarkerSize',8,'LineWidth',1.5);
hold off

grid on;
set(gca,'FontWeight','bold','LineWidth',2)
xlabel('x')
ylabel('t')
% export_fig p2_order40 -png -transparent -r150

% Maximum Likelihood:
w_MLE = (X'*X) \ X' * t;
beta_MLE = (N / ((X*w_MLE - t)'*(X*w_MLE - t)));
err_MLE = exp(-0.5*beta_MLE*((X*w_MLE - t)'*(X*w_MLE - t)) +...
   0.5*N*log(beta_MLE) - 0.5*N*log(2*pi));
Yn_MLE = AA * w_MLE;

figure(4)
hold on
shadedErrorBar(x,y,err,{'b-','color','r','LineWidth',2},0);
plot(aa, Yn_MLE, 'g', 'LineWidth', 2);
plot(x,t,'bo','MarkerSize',8,'LineWidth',1.5);
hold off

grid on;
set(gca,'FontWeight','bold','LineWidth',2)
xlabel('x')
ylabel('t')
% export_fig p3_order40 -png -transparent -r150

% Maximum a posteriori
beta_MAP = floor(beta_MLE/2);
alpha = lamda * beta_MAP;
w_MAP = (alpha * eye(M+1) + X'*X) \ X' * t;

Yn_MAP = AA * w_MAP;
err_MAP = 0.5*beta_MAP * (X*w_MAP-t)'*(X*w_MAP-t) + 0.5*alpha*w_MAP'*w_MAP;

figure(5)
hold on
shadedErrorBar(x,y,err,{'b-','color','r','LineWidth',2},0);
plot(aa, Yn_MAP, 'g', 'LineWidth',2);
plot(x,t,'bo','MarkerSize',8,'LineWidth',1.5);
hold off

grid on;
set(gca,'FontWeight','bold','LineWidth',2)
xlabel('x')
ylabel('t')
%  export_fig p4_order40 -png -transparent -r150