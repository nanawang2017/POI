%RankGeoFM methods
  %filename is filefold containing input data, 
  %K is the number of dimensions of latent space
  %k_1 is the number of neighbors considered for Geographical influence
% e.g.:   RankGeoFM('SG',100,100)


function [] = RankGeoFM2(filename, K, k_1, connect)
   inputFile=sprintf('%s/%s',filename,'train_tensor.txt');
   data=dlmread(inputFile);
   inputFile=sprintf('%s/%s',filename,'tensor_size.txt');
   tsize=dlmread(inputFile);
   
   B=sparse(data(:,1),data(:,2),data(:,3),tsize(1),tsize(2));
   [index_1,index_2,value]=find(B);
   data=[index_1,index_2,value];
   [myRow,~]=size(data);
   myShuffle=randperm(myRow);
   data=data(myShuffle,:);
   
   maginal=0.3;
   C=1;
   
   alpha=0.0001;
   iteration=0;
   FactorInf=0.2; %0.5 0.7 0.9
   a_best=0;
   
   if connect == 0
       U_1=0.01*randn(tsize(1),K); %user preference
       U_2=0.01*randn(tsize(1),K); %user preference from geographical influence
       L_1=0.01*randn(tsize(2),K);

       inputFile=sprintf('%s/%s',filename,'dist.mat');
       load(inputFile);
       inputFile=sprintf('%s/%s',filename,'Index_dist.mat');
       load(inputFile);

       [myRow,~]=size(A);
       for i=1:myRow
           x=find(A(i,:)<0.5);
           if(isempty(x) == 0)
                A(i,x)=0.5;
           end
           A(i,:)=(1./A(i,:))/sum((1./A(i,:)));
       end

       inputFile=sprintf('%s/%s',filename,'decomp-WARP.mat');
       save(inputFile,'U_1','U_2','L_1','A','Index_A');  
   else
       inputFile=sprintf('%s_POI_Rec_Iterations/decomp-WARP.mat',filename);
       load(inputFile,'U_1','U_2','L_1','A','Index_A');  
   end
   
   LossWeight=zeros(tsize(2),1);
   myTotal=0;
   for i=1:tsize(2)
       LossWeight(i)=myTotal+1/i;
       myTotal=myTotal+1/i;
   end
   
   fprintf('\nbuilding map');
   tic;
   myMap=zeros(tsize(1), tsize(2));
   for i=1:length(data)
       uid = data(i,1);
       lid = data(i,2);
       freq = data(i,3);
       myMap(uid, lid) = freq;
   end
   toc;
   
   converge = 0;
   while(converge < 1000)
       tic;
        for i=1:tsize(1)
           if(norm(U_2(i,:))>C*FactorInf)
               U_2(i,:)=FactorInf*C*U_2(i,:)/norm(U_2(i,:));
           end
        end

        F_G=zeros(tsize(2),K);
        for i=1:tsize(2)
            for j=1:k_1
                F_G(i,:)=F_G(i,:)+A(i,j)*L_1(Index_A(i,j),:);
            end
        end
        [a,b]=test_performance_new(filename, U_1, L_1, U_2, F_G, 5);
        %myOutPut=sprintf('%s_POI_Rec_Iterations/%d.mat',filename,iteration);
        %save(myOutPut,'a','b');
        if(a>a_best)
             a_best=a;
             inputFile=sprintf('%s_POI_Rec_Iterations/decomp-WARP.mat',filename);
             save(inputFile,'U_1','U_2','L_1','A','Index_A');
             converge = 0;
        else
            converge = converge + 1;
        end 
        
      U_1_pre=U_1;
      U_2_pre=U_2;
      L_1_pre=L_1;
      
      UL = U_1 * L_1';
      UFG = U_2 * F_G';
      
      for i=1:length(data)
           index_1=data(i,:);
           myu=index_1(1);
           mya=index_1(2);
           rand_nums = rand(1, tsize(2)) * tsize(2);
           [N, myb] = get_N(UL, UFG, myMap, rand_nums, tsize(1), tsize(2), myu - 1, mya - 1, maginal);
           if(N>tsize(2)-1)
               continue;
           end
           N=floor((tsize(2)-1)/N);
               delta=LossWeight(N);
               U_1(myu,:)=U_1(myu,:)+alpha*(delta*(L_1(mya,:)-L_1(myb,:)));
               U_2(myu,:)=U_2(myu,:)+alpha*(delta*(F_G(mya,:)-F_G(myb,:)));
               L_1(mya,:)=L_1(mya,:)+alpha*(delta*(U_1(myu,:)));
               L_1(myb,:)=L_1(myb,:)+alpha*(-delta*(U_1(myu,:)));
               %project to norm constraits
               if(norm(U_1(myu,:))>C)
                   U_1(myu,:)=C*U_1(myu,:)/norm(U_1(myu,:));
               end
               if(norm(U_2(myu,:))>C*FactorInf)
                   U_2(myu,:)=C*U_2(myu,:)/norm(U_2(myu,:));
               end
               if(norm(L_1(mya,:))>C)
                   L_1(mya,:)=C*L_1(mya,:)/norm(L_1(mya,:));
               end
               if(norm(L_1(myb,:))>C)
                   L_1(myb,:)=C*L_1(myb,:)/norm(L_1(myb,:));
               end
      end
      toc;
       iteration=iteration+1;
       diff=norm(U_1_pre-U_1)+norm(U_2_pre-U_2)+norm(L_1_pre-L_1);
       fprintf('\niteration=%d,diff=%f\n',iteration,diff);
       if(iteration>1000)
           break;
       end    
         inputFile=sprintf('%s/%s',filename,'decomp-WARP.mat');
         save(inputFile,'U_1','U_2','L_1','A','Index_A');
   end
         inputFile=sprintf('%s/%s',filename,'decomp-WARP.mat');
         save(inputFile,'U_1','U_2','L_1','A','Index_A');
   
end

