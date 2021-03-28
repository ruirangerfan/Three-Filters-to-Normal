%% Source code is provided by Rui Ranger Fan (www.ruirangerfan.com)
%  Paper: Three-filters-to-normal: an accurate and ultrafast surface normal estimator
%  Published on IEEE Robotics and Automation Letters (RA-L)
%  Contact: rui.fan@ieee.org

clear all; 
close all; 
clc;

%% Parameter Selection and Initialization
third_filter = 'median';
kernel_size= 3; % 3x3 kernels
dataset_name = 'torusknot';
[K, ~] = get_dataset_params([dataset_name,'/']);
% K is the camera intrinsic matrix 
[Gx, Gy] = set_kernel('fd', kernel_size); 
% create two kernels for nx and ny estimation

%% Load Data
[X, Y, Z, N, mask, umax, vmax, ~, ~]... 
= load_data([dataset_name,'/'], 1, K, kernel_size);
% X, Y, Z are the x-, y-, and z- coordinates
% mask indicates the background and foreground
% umax --> image width, vmax --> image height

nx_gt = N(:,:,1); 
ny_gt = N(:,:,2); 
nz_gt = N(:,:,3);
% ground-truth nx, ny, nz
ngt_vis = visualization_map_creation(-nx_gt, -ny_gt, -nz_gt);
% create a visualization map

%% 3F2N
D = 1./Z; % inverse depth or disparity 
Gv = conv2(D, Gy, 'same');
Gu = conv2(D, Gx, 'same');        
nx_t = Gu*K(1,1);
ny_t = Gv*K(2,2);
% estimated nx and ny
nz_t_volume = zeros(vmax, umax, 8);
% create a volume to compute nz                

for j = 1:8
    [X_d, Y_d, Z_d] = delta_xyz_computation(X, Y, Z, j);
    nz_j = -(nx_t.*X_d+ny_t.*Y_d)./Z_d;
    nz_t_volume(:,:,j) = nz_j;
end

if strcmp(third_filter, 'median')
    nz_t = nanmedian(nz_t_volume, 3);
else
    nz_t = nanmean(nz_t_volume, 3);
end

nx_t(isnan(nz_t))=0;
ny_t(isnan(nz_t))=0;
nz_t(isnan(nz_t))=-1;
% process infinite points

[nx_t, ny_t, nz_t] = ...
      vector_normalization(nx_t, ny_t, nz_t);
% normalize the estimated surface normal
nt_vis = visualization_map_creation(nx_t, ny_t, nz_t);
% create a visualization map
[error_map, ea, ep] = ...
evaluation(nx_gt, ny_gt, nz_gt, nx_t, ny_t, nz_t, mask, vmax, umax, 30);
% evaluation, ea and ep are explained in the paper

%% Visualization
fig = figure('units','normalized','outerposition',[0 0 0.5 0.5]);
ax1 = subplot(2,2,1); 
imshow(ngt_vis);
title('Ground truth')
ax2 = subplot(2,2,2); 
imshow(nt_vis);
title('Result')
ax3 = subplot(2,2,[3,4]); 
imshow(error_map, [],'Colormap',jet(4096)); 
title('Error map (degrees)')
c = colorbar;
c.Label.String = 'error (degrees)';





%% additional functions

function [K, max_frm] = get_dataset_params(dataset_name)
    fileID = fopen(['./', dataset_name,'/params.txt'],'r');
    tmp = fscanf(fileID,'%f');
    fclose(fileID);
    fx = tmp(1); 
    fy = tmp(2);
    uo = tmp(3);
    vo = tmp(4);
    max_frm = tmp(5);
    K = [fx, 0, uo; 0, fy, vo; 0, 0, 1];
end

function [X, Y, Z, N, mask, umax, vmax, u_map, v_map] = load_data(dataset_name, frm, K, kernel_size)
    umax = 640; vmax = 480;
    N = double(imread([dataset_name, '/normal/',sprintf('%06d',frm),'.png']))/65535*2-1;
    file_name = [dataset_name, '/depth/',sprintf('%06d',frm),'.bin'];
    file_id = fopen(file_name);
    data = fread(file_id, 'float');
    fclose(file_id);

    u_map = ones(vmax,1)*[1:umax] -K(1,3);
    v_map = [1:vmax]'*ones(1,umax)-K(2,3);
    Z = reshape(data, umax, vmax)';
    X = u_map.*Z/K(1,1);
    Y = v_map.*Z/K(2,2);
    
    %% create mask
    mask = zeros(vmax, umax);
    mask(Z==1) = 1; 
    mask(:,1:(kernel_size-1)/2) = 1; 
    mask(1:(kernel_size-1)/2,:) = 1;
    mask(:,end+1-(kernel_size-1)/2:end) = 1; 
    mask(end+1-(kernel_size-1)/2:end,:) = 1;
    
    nx_gt = N(:,:,1); ny_gt = N(:,:,2); nz_gt = N(:,:,3);
    [nx_gt, ny_gt, nz_gt] = vector_normalization(nx_gt, ny_gt, nz_gt);
    N(:,:,1) = nx_gt; N(:,:,2) = ny_gt; N(:,:,3) = nz_gt;
end

function [nx,ny,nz]=vector_normalization(nx,ny,nz)
    mag=sqrt(nx.^2+ny.^2+nz.^2);
    nx=nx./mag;
    ny=ny./mag;
    nz=nz./mag;
end

function n_vis=visualization_map_creation(nx,ny,nz)
    [vmax, umax] = size(nx);
    n_vis = zeros(vmax,umax,3);
    n_vis(:,:,1) = nx;
    n_vis(:,:,2) = ny;
    n_vis(:,:,3) = nz;
    n_vis = (1-n_vis)/2;
end

function [X_d, Y_d, Z_d] = delta_xyz_computation(X,Y,Z,pos)

    if pos==1
        kernel=[0,-1,0;0,1,0;0,0,0];
    elseif pos==2
        kernel=[0,0,0;-1,1,0;0,0,0];
    elseif pos==3
        kernel=[0,0,0;0,1,-1;0,0,0];
    elseif pos==4
        kernel=[0,0,0;0,1,0;0,-1,0];
    elseif pos==5
        kernel=[-1,0,0;0,1,0;0,0,0];
    elseif pos==6
        kernel=[0,0,0;0,1,0;-1,0,0];
    elseif pos==7
        kernel=[0,0,-1;0,1,0;0,0,0];
    else
        kernel=[0,0,0;0,1,0;0,0,-1];
    end

    X_d = conv2(X, kernel, 'same');
    Y_d = conv2(Y, kernel, 'same');
    Z_d = conv2(Z, kernel, 'same');

    X_d(Z_d==0) = nan;
    Y_d(Z_d==0) = nan;
    Z_d(Z_d==0) = nan;

end

function angle_map = angle_normalization(angle_map)
    for i=1:numel(angle_map)
        if angle_map(i)>pi/2
            angle_map(i)=pi-angle_map(i);
        end
    end
end

function [Gx, Gy] = set_kernel(kernel_name, kernel_size)
    if strcmp(kernel_name, 'fd')
        Gx = zeros(kernel_size);
        mid = (kernel_size + 1) / 2;
        temp = 1;
        for index = (mid+1) : kernel_size
            Gx(mid, index) = temp;
            Gx(mid, 2*mid-index) = -temp;
            temp = temp + 1;
        end
        Gy = Gx';
    elseif strcmp(kernel_name, 'sobel') | strcmp(kernel_name, 'scharr') | strcmp(kernel_name, 'prewitt')
        if strcmp(kernel_name, 'sobel')
            smooth = [1 2 1];
        elseif strcmp(kernel_name, 'scharr')
            smooth = [1 1 1];
        else
            smooth = [3 10 3];
        end
        kernel3x3 = smooth' * [-1 0 1];
        kernel5x5 = conv2(smooth' * smooth, kernel3x3);
        kernel7x7 = conv2(smooth' * smooth, kernel5x5);
        kernel9x9 = conv2(smooth' * smooth, kernel7x7);
        kernel11x11 = conv2(smooth' * smooth, kernel9x9);
        if kernel_size == 3
            kernel = kernel3x3; Gx = kernel; Gy = Gx';
        elseif kernel_size == 5
            kernel = kernel5x5; Gx = kernel; Gy = Gx';
        elseif kernel_size == 7
            kernel = kernel7x7; Gx = kernel; Gy = Gx';
        elseif kernel_size == 9
            kernel = kernel9x9; Gx = kernel; Gy = Gx';
        elseif kernel_size == 11
            kernel = kernel11x11; Gx = kernel; Gy = Gx';
        end
    end
end

function [error_map, ea, ep] = evaluation(nx_gt, ny_gt, nz_gt, nx, ny, nz, mask, vmax, umax, tr)
    scale = pi/180;
    error_map = acos(nx_gt.*nx+ny_gt.*ny+nz_gt.*nz);
    error_map = angle_normalization(error_map);
    error_map(mask==1) = nan;
    error_vector = reshape(error_map/scale,[vmax*umax,1]);
    error_vector(isnan(error_vector))=[];
    ea=mean(error_vector);
    ep = [];
    for j = tr:-5:5
    tmp = error_vector;
    tmp(error_vector > j) = [];
    ep_tmp = length(tmp) / length(error_vector);
    ep = [ep; ep_tmp];
    end
end
