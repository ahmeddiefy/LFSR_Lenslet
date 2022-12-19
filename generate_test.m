clearvars; clearvars -global; clc; close all; warning off;
datasetFolder = './datasets/';
angRes = 5;

extension = [];
list = dir([datasetFolder, '/*', extension]);
datasetNames = setdiff({list.name}, {'.', '..'});

for ds = 1:length(datasetNames)
    folderPath = strcat(datasetFolder,datasetNames{ds},'/test/');
    mkdir(strcat('./test/2xSR/',datasetNames{ds}));
    mkdir(strcat('./test/4xSR/',datasetNames{ds}));
    
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

        img_raw = single(zeros( U*V, h, w));
        img_2 = single(zeros( U*V, h, w));
        img_4 = single(zeros( U*V, h, w));

        for ax = 1 : 5
            for ay = 1 : 5

                step_image  = rgb2ycbcr(double(squeeze(LF_image(ay, ax, :, :, :))));
                step_image = step_image(:,:,1);      
                img_raw( sub2ind([5 5], ax, ay), :, :) = step_image; 
                img_2( sub2ind([5 5], ax, ay), :, :) = imresize(imresize(step_image, [floor(h/2) floor(w/2)]), [h w]);
                img_4( sub2ind([5 5], ax, ay), :, :) = imresize(imresize(step_image, [floor(h/4) floor(w/4)]), [h w]);
            end
        end
       s_name=sceneNames{ns};
       patch_name_2x = strcat('./test/2xSR/',datasetNames{ds},'/',s_name(1:end-4));
       patch_name_4x = strcat('./test/4xSR/',datasetNames{ds},'/',s_name(1:end-4));

       save(patch_name_2x, 'img_raw');
       save(sprintf('%s_2', patch_name_2x), 'img_2');
       
       save(patch_name_4x, 'img_raw');
       save(sprintf('%s_4', patch_name_4x), 'img_4');

       display(ns);


    end
end
