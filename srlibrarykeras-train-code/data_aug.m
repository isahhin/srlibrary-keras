
clear
close all;
folder = 'DIV2K';
savepath = 'DIV2K-aug/';

filepaths = dir(fullfile('DIV2K', '*.png'));
     
for i = 1 : length(filepaths)
    filename = filepaths(i).name;
    [add, im_name, type] = fileparts(filepaths(i).name);
    %im_name
    image = imread(fullfile(folder, filename));
    image = imresize(image, 0.50, 'bicubic');
    %imwrite(image, [savepath im_name, '.png']);
    for angle = 1:2
       
        im_rot = rot90(image, angle);
        imwrite(im_rot, [savepath im_name, '_rot' num2str(angle*90) '.png']);
        
        im_flip_h = fliplr(im_rot);
        imwrite(im_flip_h, [savepath im_name, '_flip_h' num2str(angle*90) '.png']);
        
%         for scale = 0.6 : 0.1 :0.7
%             im_down = imresize(im_rot, scale, 'bicubic');
%             imwrite(im_down, [savepath im_name, '_rot' num2str(angle*90) '_s' num2str(scale*10) '.png']);
%         end
        
        
    end
end