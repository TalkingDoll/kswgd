% this is a Gaussian example as a proof of concept. 
clear 
close all

%% we first sample 5000 points from a 2D Guassian mixture 
% 2/5*N(-3,1) + 1/5*N(0,1) + 2/5*N(4,2)
n = 500;

d = 100;
lambda = 1;
u = normrnd(0,1,[n,d]);
u(:,1) = lambda * u(:,1);
u_norm = sqrt(sum(u.^2, 2));
r = sqrt(rand(n,1)) * 1/100 + 99/100;
u_trans = u ./u_norm;

% u_trans(u_trans(:,3)>0.5 && u_trans(:,3)<0.6)
X_tar = r.* u_trans;
% X_tar = X_tar(X_tar(:,2)>0,:);
% X_tar = X_tar;

n = size(X_tar,1);






%% then form the anisotropic graph laplacian
sq_tar = sum(X_tar.^2,2);
H = (sq_tar' + sq_tar - 2 * (X_tar * X_tar'));
epsilon = 1/2*median(H(:)) / log(n+1); 
% epsilon = 0.18;
sq_tar = sum(X_tar.^2,2);
ker = @(X) exp(-(sq_tar' + sq_tar - 2 * (X_tar * X_tar')) ./ (2*epsilon));
%  

data_kernel = ker(X_tar);
% calculate sum of rows, which is equal to sum of columns.
p_x = sqrt(sum(data_kernel))';
p_y = p_x';

% then normalize
data_kernel_norm = data_kernel ./ p_x ./ p_y;
D_y = sum(data_kernel_norm);


rw_kernel = 1/2 * (data_kernel_norm ./ D_y + data_kernel_norm ./ D_y');
[phi, lambda_ns] = svd(rw_kernel);
lambda = -diag(lambda_ns) + 1;
inv_lambda = [0; 1 ./ lambda(2:end)] * epsilon;
inv_K = phi * diag(inv_lambda) * phi';




tol = 1e-6;
lambda_ns = diag(lambda_ns);
lambda_ns(lambda_ns<tol) = 0;
below_tol = length(lambda_ns(lambda_ns<tol)); 
above_tol = n - below_tol;
reg = .001;
lambda_ns_inv = epsilon * [1./(lambda_ns(lambda_ns>=tol)+reg); zeros(below_tol, 1)];

inv_K_ns = phi * diag(lambda_ns_inv) * phi';


% diag(lambda_ns_inv) * phi' * test1;






%% run algorithm

% initialize particles

iter = 1000;
h = 20;


m = 700;


u = normrnd(0,1,[m,d]);
u_norm = sqrt(sum(u.^2, 2));
r = sqrt(rand(m,1)) * 1/100 + 99/100;
u_trans = u ./u_norm;
% u_trans(u_trans(:,3)>0.5 && u_trans(:,3)<0.6)
x_init = r.* u_trans;
x_init = x_init(x_init(:,2)>0.2,:);
% x_init = mvnrnd([0,0],eye(2)*0.01, m);
m = size(x_init,1);

x_t = zeros(m, d,iter);


x_t(:,:,1) = x_init;




p_tar = sum(data_kernel)';
D = sum(data_kernel ./ sqrt(p_tar) ./ sqrt(p_tar)',2);

inv_K_ns_s_ns = phi * diag(lambda_ns_inv .* inv_lambda .* lambda_ns_inv) * phi'; 

lambda_s_s_ns = inv_lambda .* inv_lambda .* lambda_ns_inv;
lambda_s_s_ns = lambda_s_s_ns(1:above_tol);

lambda_ns_s_ns = lambda_ns_inv .* inv_lambda .* lambda_ns_inv;
lambda_ns_s_ns = lambda_ns_s_ns(1:above_tol);

%%

sum_x = zeros(m,d);
for t = 1:iter-1

%     if mod(t,100)==0
%         fprintf('iter %d\n', t)
%     end


    grad_matrix = grad_ker1(x_t(:,:,t), X_tar, p_tar,sq_tar, D, epsilon);
    cross_matrix = K_tar_eval(X_tar, x_t(:,:,t), p_tar, sq_tar, D, epsilon);
    
    
    for i = 1:d
        sum_x(:,i) = sum(grad_matrix(:,:,i) * phi(:,1:above_tol) * diag(lambda_ns_s_ns) * phi(:,1:above_tol)' * cross_matrix,2);
%         sum_x(:,i) = sum_x(:,i)./norm(sum_x(:,i));
%         sum_x(:,i) = sum(grad_matrix(:,:,i) *  cross_matrix,2);
    end
  
     
    x_t(:,:,t+1) = x_t(:,:,t) - h/m * sum_x;
        
   
end




%%
if d==2
    figure;
    plot(X_tar(:,1), X_tar(:,2), 'o');
    hold on
    plot(x_t(:,1,1),x_t(:,2,1),'o');
    hold on
    plot(x_t(:,1,t), x_t(:,2, t), 'o');
else

    figure;
    scatter3(X_tar(:,1), X_tar(:,2), X_tar(:,3), 'o');
    hold on
    scatter3(x_t(:,1,1),x_t(:,2,1),x_t(:,3, 1),'o');
    hold on
    scatter3(x_t(:,1,t), x_t(:,2, t), x_t(:,3, t),'o');
    figure; 
    scatter3(x_t(:,1,1),x_t(:,2,1),x_t(:,3, 1),'o');
    hold on
    scatter3(x_t(:,1,end), x_t(:,2, end), x_t(:,3, end),'o');
end


%% plot matrix
figure; plotmatrix(X_tar); 
figure; plotmatrix(x_t(:,:,end))


%% Save figures
scriptFolder = fileparts(mfilename('fullpath'));

saveFolder = fullfile(scriptFolder, 'figure_dmps_test');

if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

figHandles = findall(0, 'Type', 'figure');

for k = 1:length(figHandles)
    fig = figHandles(k);
    filename = fullfile(saveFolder, sprintf('figure_%d.png', k));
    saveas(fig, filename);
end
