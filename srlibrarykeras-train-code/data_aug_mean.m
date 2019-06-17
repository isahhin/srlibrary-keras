
clear
close all;
folder = 'DIV2K';
savepath = 'DIV2K-aug/';

filepaths = dir(fullfile('DIV2K', '*.png'));
     
m_y=0;
for i = 1 : length(filepaths)
    filename = filepaths(i).name;
    [add, im_name, type] = fileparts(filepaths(i).name);
    %im_name
    image = imread(fullfile(folder, filename));
    YCbCr = rgb2ycbcr(image);
    YCbCr = YCbCr(:,:,1);
    m = mean( YCbCr(:));
    m_y = m_y + m;
end
m_y = m_y/800