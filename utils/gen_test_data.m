clc
clear
%% paths (Please specify your data path)
path_src  = 'F:/LongguangWang/Data/benchmark/Set5/HR';
path_save = 'F:/LongguangWang/Data/benchmark/Set5/';
ext       =  {'*.png'};
HR_path   =  [];
LR_path = fullfile(path_save,'LR_bicubic');
if ~exist(LR_path)
    mkdir(LR_path)
end

for i = 1:length(ext)
    HR_path = cat(1,HR_path, dir(fullfile(path_src, ext{i})));
end
n_imgs = length(HR_path);

%% read HR images
DIV2K_HR = [];
for idx_im = 1:n_imgs
    fprintf('Read HR :%d\n', idx_im);
    ImHR = imread(fullfile(path_src, HR_path(idx_im).name));
    DIV2K_HR{idx_im} = ImHR;
end

%% generate LR images (Please specify scale factors you want to evaluation)
scale_list = [2, 2,5, 3.1];
scale2_list = [2.5, 2,5, 3.1];
n_scale = length(scale_list);

for idx_im = 1:n_imgs
    fprintf('IdxIm=%d\n', idx_im);
    ImHR = DIV2K_HR{idx_im};
    
    for i = 1:1:n_scale
        scale = scale_list(i);
        scale2 = scale2_list(i);
        
        if scale == scale2
            LR_path_son = fullfile(path_save, 'LR_bicubic', sprintf('X%.2f',scale));
        else
            LR_path_son = [path_save,'/LR_bicubic/', sprintf('X%.2f',scale), '_', sprintf('X%.2f',scale2),];         
        end
        if ~exist(LR_path_son, 'dir')
            mkdir(LR_path_son)
        end
        
        % bicubic downsampling
        ImLR = gen_LR_img(ImHR, scale, scale2);
        
        % save image
        img_name = HR_path(idx_im).name;
        ImLR_name = fullfile(LR_path_son, img_name);
        imwrite(ImLR, ImLR_name, 'png');
    end
end