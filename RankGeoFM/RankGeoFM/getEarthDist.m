function [ dist ] = getEarthDist(lat1,lng1,lat2,lng2)
    radLat1=rad(lat1);
    radLat2=rad(lat2);
    a=radLat1-radLat2;
    b=rad(lng1)-rad(lng2);
    
    s=2*asin(sqrt(power(sin(a/2),2)+cos(radLat1)*cos(radLat2)*power(sin(b/2),2)));
    s=s*6371.004;
  %  s=round(s*10000)/10000;
    dist=s;
end

