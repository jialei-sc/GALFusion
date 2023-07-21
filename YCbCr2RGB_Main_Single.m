clc;
clear all;
A_dir = fullfile('D:\matlab_utils\Evaluation-for-Image-Fusion-main\Evaluation-for-Image-Fusion-main\Image\Source-Image\CT_MRI\ct'); % 包含所有源图像A的文件夹路径
Fused_dir = fullfile('D:\MT\model_pth\origi-train-network\base-MT-v1\autodl-tmp\base-MT-old\Multi_task_result\MT_e500_len_after_p_1_wo_spectral\ct_mri'); % 融合图像的Y通道的文件夹路径
save_dir = fullfile('D:\matlab_utils\Evaluation-for-Image-Fusion-main\Evaluation-for-Image-Fusion-main\Image\Algorithm\MT-exp-abla-sdl\CT_MRI\wo_spectral\'); % 彩色融合图像的文件夹路径
% M3FD   D:\MT数据\MT_test_dataset\M3FD\M3FD_Fusion\ir_vis_set_M3FD\M3FD_vis
% MSRS   D:\数据集\MSRS\msrs-data\fusion_vis
% SPECT  D:\MT数据\MT_test_dataset\test_Medical\MRI-SPECT\spect_mri_set\spect
% PET    D:\MT数据\MT_experiments_results\test_pet
% CT     D:\matlab_utils\Evaluation-for-Image-Fusion-main\Evaluation-for-Image-Fusion-main\Image\Source-Image\CT_MRI\ct
% M3FD-detect D:\数据集\M3FD\M3FD_Detection\vi
% MSRS-seg D:\数据集\MSRS\MSRS\MSRS-data\vi


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