function [] = getDist(filename, nk)
    locationFile = sprintf('%s/%s',filename,'tensor_lat_lng.txt');
    locations = dlmread(locationFile);
    sizeFile = sprintf('%s/%s',filename,'tensor_size.txt');
    tsize = dlmread(sizeFile);
    row = tsize(2);
    Index_A = zeros(row, nk);
    A = zeros(row, nk);
    dist = zeros(1, row);
    for i=1:row
        if mod(i, 10) == 0
            disp(i);
        end
        for j=1:row
            if i == j
                dist(1, j) = 99999999;
            else
                dist(1, j) = getEarthDist(locations(i,1),locations(i,2),locations(j,1),locations(j,2));
            end
        end
        [x,y] = sort(dist(1,:),'ascend');
        A(i, :) = x(1:nk);
        Index_A(i, :) = y(1:nk);
    end
    outputFile = sprintf('%s/%s',filename,'dist.mat');
    save(outputFile,'A');
    outputFile = sprintf('%s/%s',filename,'Index_dist.mat');
    save(outputFile,'Index_A');
end
