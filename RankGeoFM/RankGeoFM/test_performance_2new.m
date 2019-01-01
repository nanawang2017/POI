function [All_pre, All_recall] = test_performance_2new( filename, flagTest, N, flagMethod )

   %flagMethod =1 CF, =2 CF+G, =3 GeoMF 
   alpha=0.7;
   inputFile=sprintf('%s/%s',filename,'train_tensor.txt');
   data_known=dlmread(inputFile);
   inputFile=sprintf('%s/%s',filename,'tensor_size.txt');
   tsize=dlmread(inputFile);
   
   if(flagTest)
       inputFile=sprintf('%s/%s',filename,'test_tensor.txt');
   else      
       inputFile=sprintf('%s/%s',filename,'tune_tensor.txt');
   end
   data_pred=dlmread(inputFile);
   if(flagMethod==1)
       inputFile=sprintf('%s/%s',filename,'CF.mat');
       load(inputFile);
   elseif(flagMethod==2)
       inputFile=sprintf('%s/%s',filename,'CF+G.mat');
       load(inputFile);
   elseif(flagMethod==3)
       inputFile=sprintf('%s/%s',filename,'GeoMF.mat');
       load(inputFile);
       PQ=P*Q';
       XY=X*Y';
   elseif(flagMethod==4)
       inputFile=sprintf('%s/%s',filename,'decomp-WARP.mat');
       load(inputFile);
        [~,K]=size(U_1);
        [~,k_1]=size(Index_A);
        if(k_1>0)
            F_G=zeros(tsize(2),K);
            for i=1:tsize(2)
                for j=1:k_1
                    F_G(i,:)=F_G(i,:)+A(i,j)*L_1(Index_A(i,j),:);
                end
            end
        end
            UL=U_1*L_1';
            UFG=U_2*F_G';
   elseif(flagMethod==5)
       inputFile=sprintf('%s/%s',filename,'decomp-BPR.mat');
       load(inputFile);
        [~,K]=size(U_1);
        [~,k_1]=size(Index_A);
        if(k_1>0)
            F_G=zeros(tsize(2),K);
            for i=1:tsize(2)
                for j=1:k_1
                    F_G(i,:)=F_G(i,:)+A(i,j)*L_1(Index_A(i,j),:);
                end
            end
        end
            UL=U_1*L_1';
            UFG=U_2*F_G';
   elseif(flagMethod==6)
       inputFile=sprintf('%s/%s',filename,'GTBNMF.mat');
       load(inputFile)
       M=U*V';
      M=M*spdiags(1./(0.2+dist).*roh,0,tsize(2),tsize(2));
%        for i=1:tsize(2)
%            M(:,i)=M(:,i)*(1/(0.2+dist(i)))*roh(i);
%        end
   elseif(flagMethod==7)
       inputFile=sprintf('%s/%s',filename,'GeoPMF.mat');
       load(inputFile)
       load(inputFile)
       M=U*V';
       M=M*spdiags(0.2./(0.2+dist),0,tsize(2),tsize(2));
%        for i=1:tsize(2)
%            M(:,i)=M(:,i)*(0.2/(0.2+dist(i))*roh(i));
%        end

   end
  for i=1:tsize(2)    
      if(i<20)
        pos_2_scores(i)=1/(i+1);
      else
        pos_2_scores(i)=1/(2*(i+1));
      end
  end
   
   All_pre=cell(length(N),1);
   All_recall=cell(length(N),1);
   for k=1:length(N)
       All_pre{k}=0;
       All_recall{k}=0;
   end

      %calculate the precision and recall for each hour
      %if there is no testing data for a particular hour, 
      %the precision and recall are set as -1;
          subdata=data_pred;
          users=unique(subdata(:,1));
          %record the locations that have been visited in training data
          location_visited=[];
          for u=1:length(users)
              location_visited=unique(data_known(find(data_known(:,1)==users(u)),2));
              index=find(subdata(:,1)==users(u));
              target_locations=unique(subdata(index,2));       
   
              if(flagMethod==1)
                 scores=Rec_S(users(u),:);   
              elseif(flagMethod==2)
                 x=Rec_S(users(u),:);
                 y=Rec_G(users(u),:);
                 x(location_visited)=0;
                 y(location_visited)=0;
                 [~,index_x]=sort(x,'descend');
                 [~,index_y]=sort(y,'descend');
                 x(index_x)=pos_2_scores;
                 scores=alpha*x+(1-alpha)*y;
              elseif(flagMethod==3)
                 scores=PQ(users(u),:)+XY(users(u),:);
                % scores=PQ(users(u),:);
              elseif(flagMethod==4||flagMethod==5)
                 scores=zeros(tsize(2),1);
                 for j=1:tsize(2)
                     %scores(j)=sum(U_1(users(u),:).*L_1(j,:))+sum(U_2(users(u),:).*F_G(j,:));
                     scores(j)=UL(users(u),j)+UFG(users(u),j);
                 end
              elseif(flagMethod==6)
                  scores=M(users(u),:);
              elseif(flagMethod==7)
                  scores=M(users(u),:);
              end
              
              myTemp=scores(target_locations);
              scores(location_visited)=-1;
              scores(target_locations)=myTemp;
              [a, b]=sort(scores,'descend');
              tp=zeros(length(N),1);
              tn=zeros(length(N),1);
              fp=zeros(length(N),1);
              for k=1:length(N)
                      for j=1:N(k)
                          if(numel(find(target_locations==b(j)))>0)
                              tp(k)=tp(k)+1;
                          else
                              fp(k)=fp(k)+1;
                          end
                      end
                      for j=1:length(target_locations)
                          if(numel(find(b(1:N(k))==target_locations(j)))==0)
                              tn(k)=tn(k)+1;
                          end
                      end
                   All_pre{k}=All_pre{k}+tp(k)/(tp(k)+fp(k));
                   All_recall{k}=All_recall{k}+tp(k)/(tp(k)+tn(k));
              end
          end
          for k=1:length(N)  
                pre=All_pre{k}/length(users);
                recall=All_recall{k}/length(users);
                fprintf('@%d,pre=%f,recall=%f\n',N(k),pre,recall);
                All_pre{k}=pre;
                All_recall{k}=recall;
          end


end

