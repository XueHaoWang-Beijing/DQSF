function imgn = prog_tr(img)
%PROG_TR 此处显示有关此函数的摘要
%   此处显示详细说明
    [x1 x2 x3] = size(img);
    img = imresize(img,[300,300]);
    [m n ~]=size(img);  
    ruler = [3,7];
    imgn=zeros(m,n);
    for num = 1:1
        w=ruler(num);    %模板半径 模板大小为9*9  
        for i=1+w:m-w  
            for j=1+w:n-w  
                Hist=zeros(1,256);  
                for p=i-w:i+w  
                    for q=j-w:j+w  
                        Hist(img(p,q)+1)=Hist(img(p,q)+1)+1;    %统计局部直方图  
                    end  
                end  
                Hist=Hist/sum(Hist);     %部分人称之为归一化直方图  
                for k=1:256  
                    if Hist(k)~=0  
                       imgn(i,j,num)=imgn(i,j)+Hist(k)*log(1/Hist(k));  %局部熵  
                    end  
                end  
            end  
        end  
    end
    imgn = imresize(imgn,[x1,x2]);
%     imshow(imgn,[])  
end

