clearvars; clearvars -global; clc; close all; warning off;
folderPath = './Training';
mkdir('train');
patch_size = 32; 
angRes = 5;
count = 1;

SR_factor = 2; % 2 or 4

extension = [];
list = dir([folderPath, '/*', extension]);
sceneNames = setdiff({list.name}, {'.', '..'});
scenePaths = strcat(strcat(folderPath, '/'), sceneNames);
numScenes = length(sceneNames);

for ns = 1:numScenes
    
   img =  load(scenePaths{ns});
   image = img.LF;
   [U,V,H,W,D] = size(image);
   LF_image = image(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), :, :, 1:3);
   
   
    h = floor(H/4)*4;
    w = floor(W/4)*4;

    LF_image = LF_image(:, :, 1:h, 1:w, :);

    GroundTruth = single(zeros( U*V, h, w));
    Input = single(zeros( U*V, h, w));

    for ax = 1 : 5
        for ay = 1 : 5

            step_image  = rgb2ycbcr(double(squeeze(LF_image(ay, ax, :, :, :))));
            step_image = step_image(:,:,1);      
            GroundTruth( sub2ind([5 5], ax, ay), :, :) = step_image; 
            Input( sub2ind([5 5], ax, ay), :, :) = imresize(imresize(step_image, [floor(h/SR_factor) floor(w/SR_factor)]), [h w]);
        end
    end

    
    for ix=1:floor(h/patch_size)
        for iy=1:floor(w/patch_size)
           patch_name = sprintf('./train/%d',count);
           patch =  GroundTruth( :, (ix-1)*patch_size + 1:ix * patch_size, (iy-1)*patch_size + 1:iy * patch_size);
           save(patch_name, 'patch');
           
           patch= Input( :, (ix-1)*patch_size + 1:ix * patch_size, (iy-1)*patch_size + 1:iy * patch_size);
           save(sprintf('%s_2', patch_name), 'patch');
           
           count = count+1;

        end
    end

    
    
    
end