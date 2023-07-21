clc;
clear all;
A_dir = 'D:\MSRS-data\vi';
save_dir_A_Y = 'D:\MSRS-data\vi-y-cb-cr\Y\';
save_dir_A_Cb = 'D:\MSRS-data\vi-y-cb-cr\Cb\';
save_dir_A_Cr = 'D:\MSRS-data\vi-y-cb-cr\Cr\';
 if exist(save_dir_A_Y,'dir')==0
	mkdir(save_dir_A_Y);
 end
 if exist(save_dir_A_Cb,'dir')==0
	mkdir(save_dir_A_Cb);
 end
 if exist(save_dir_A_Cr,'dir')==0
	mkdir(save_dir_A_Cr);
 end
 fileFolder=fullfile(A_dir); 
dirOutput=dir(fullfile(fileFolder,'*.png')); % 图像后缀
fileNames = {dirOutput.name};
[m, num] = size(fileNames);
for i = 1:num
	name_A = fullfile(A_dir, fileNames{i});
    save_name_A_Y = strcat(save_dir_A_Y, fileNames{i});
    save_name_A_Cb = strcat(save_dir_A_Cb, fileNames{i});
    save_name_A_Cr = strcat(save_dir_A_Cr, fileNames{i});
    image_A = double(imread(name_A));
    [Y_A,Cb_A,Cr_A]=RGB2YCbCr(image_A); 
    imwrite(uint8(Y_A), save_name_A_Y);
    imwrite(uint8(Cb_A), save_name_A_Cb);
    imwrite(uint8(Cr_A), save_name_A_Cr);
    disp(save_name_A_Y)
end