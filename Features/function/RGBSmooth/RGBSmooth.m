function [Result] = RGBSmooth(img, sal,ran,str,num)


I= img;
[SEGMENTS, numlabels] = slicmex(im2uint8(I),num,1);%
 %[SEGMENTS, numlabels] = slicomex(im2uint8(I),800);%numlabels is the same as number of superpixels
 SEGMENTS = SEGMENTS + 1;
 SP.SuperPixelNumber = max(max(SEGMENTS));
 SP.Clustering = zeros(SP.SuperPixelNumber,2000,9);
 SP.ClusteringPixelNum = zeros(1,SP.SuperPixelNumber);
 W = size(I,1); H = size(I,2);
   for i=1:W
       for j=1:H
           SIndex = SEGMENTS(i,j);
           if(SP.ClusteringPixelNum(1,SIndex)+1<=2000)
               SP.ClusteringPixelNum(1,SIndex) = SP.ClusteringPixelNum(1,SIndex)+1;
               RGBcom = I(i,j,1:3);
%                Lcom = L(i,j,1:3);
               SP.Clustering(SIndex,SP.ClusteringPixelNum(1,SIndex),1:2) = [i j]';
               SP.Clustering(SIndex,SP.ClusteringPixelNum(1,SIndex),3:5) = RGBcom(:)';
%                SP.Clustering(SIndex,SP.ClusteringPixelNum(1,SIndex),6:8) = Lcom(:)';
               SP.Clustering(SIndex,SP.ClusteringPixelNum(1,SIndex),9) = sal(i,j);
           end
       end
   end
   SP.MiddlePoint = zeros(SP.SuperPixelNumber,9);
   for i=1:SP.SuperPixelNumber
       temp = shiftdim(SP.Clustering(i,1:SP.ClusteringPixelNum(1,i),:));
       SP.MiddlePoint(i,:)=mean(temp);
   end
   
   %Smooth
   SResult = zeros(1,SP.SuperPixelNumber);
   for i=1:SP.SuperPixelNumber
       IIndexX = SP.MiddlePoint(i,1);IIndexY = SP.MiddlePoint(i,2);
       SS = 0;
       WS = 0;
       for j=1:SP.SuperPixelNumber
          JIndexX = SP.MiddlePoint(j,1);JIndexY = SP.MiddlePoint(j,2);
          Dist = abs(IIndexX-JIndexX) + abs(IIndexY-JIndexY);
          if(Dist<ran)
              %Smooth
              ColorDist = sqrt(sum((SP.MiddlePoint(i,3:5) - SP.MiddlePoint(j,3:5)).^2));
%               LDist = sqrt(sum((SP.MiddlePoint(i,6:8) - SP.MiddlePoint(j,6:8)).^2));
              TDist = exp(-(ColorDist)*str);%%²ÎÊýÉèÖÃ
              SS = SS + TDist*SP.MiddlePoint(j,9);
              WS = WS + TDist;
          end
       end
      SResult(1,i) = SS/WS;
   end
   
   Result = zeros(300,300);
 for i=1:SP.SuperPixelNumber
     for j=1:SP.ClusteringPixelNum(1,i)
     IndeX = SP.Clustering(i,j,1);
     IndeY = SP.Clustering(i,j,2);
     Result(IndeX,IndeY) = SResult(1,i);
     end
 end