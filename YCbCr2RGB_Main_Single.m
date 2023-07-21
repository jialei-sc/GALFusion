clc;
clear all;
A_dir = fullfile(''); % 包含所有源图像A的文件夹路径
Fused_dir = fullfile(''); % 融合图像的Y通道的文件夹路径
save_dir = fullfile(''); % 彩色融合图像的文件夹路径

fileFolder=fullfile(A_dir);
fileFolder_F=fullfile(Fused_dir);
dirOutput=dir(fullfile(fileFolder,'*.png'));
dirOutput_F=dir(fullfile(fileFolder_F,'*.png'));

fileNames = {dirOutput.name};
fileNames_F = {dirOutput_F.name};
[m, num] = size(fileNames);
if exist(save_dir,'dir')==0
	mkdir(save_dir);
end
for i = 1:num
    name_A = fullfile(A_dir, fileNames{i});
    name_fused = fullfile(Fused_dir, fileNames_F{i});
    save_name = fullfile(save_dir, fileNames_F{i});    
    image_A = double(imread(name_A));
    I_result = double(imread(name_fused));
    [Y1,Cb1,Cr1]=RGB2YCbCr(image_A);
    I_final_YCbCr=cat(3,I_result,Cb1,Cr1);
    I_final_RGB=YCbCr2RGB(I_final_YCbCr);
    imwrite(uint8(I_final_RGB), save_name);
    disp(save_name);
end