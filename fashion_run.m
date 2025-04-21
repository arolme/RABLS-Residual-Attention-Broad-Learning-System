clear; 
clc;
clf;
warning off all;
format compact;
dataset_name = 'fashion_mnist';
data_path = ['./data/', dataset_name, '.mat'];
load(data_path);
rng(8148);

%Data and labels are reset
train_x = double(train_x);
train_y = double(train_y);
test_x = double(test_x);
test_y = double(test_y);
train_y=(train_y-1)*2+1;
test_y=(test_y-1)*2+1;

[Train_x, Test_x]=pre_zca(train_x,test_x);
train_x=Train_x;test_x=Test_x;
train_x = zscore(train_x')';
test_x = zscore(test_x')';

%Samples are normalized and the lable data
lambda = 2^-30; scaling_param = 0.8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
features_per_window=50;%feature nodes  per window
num_windows=30;% number of windows of feature nodes
num_enhancements=2000;% number of enhancement nodes
max_iterations=19;
try_gpu = false;

[best_accuracy, best_rmse, best_iteration, pseudoinverse_times,accuracy_history,rmse_history,residual_history] = RA_BLS(train_x,train_y,test_x,test_y,scaling_param, lambda, features_per_window, num_windows, num_enhancements, max_iterations,try_gpu);
