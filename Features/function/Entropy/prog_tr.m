function imgn = prog_tr(img)
%PROG_TR �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    [x1 x2 x3] = size(img);
    img = imresize(img,[300,300]);
    [m n ~]=size(img);  
    ruler = [3,7];
    imgn=zeros(m,n);
    for num = 1:1
        w=ruler(num);    %ģ��뾶 ģ���СΪ9*9  
        for i=1+w:m-w  
            for j=1+w:n-w  
                Hist=zeros(1,256);  
                for p=i-w:i+w  
                    for q=j-w:j+w  
                        Hist(img(p,q)+1)=Hist(img(p,q)+1)+1;    %ͳ�ƾֲ�ֱ��ͼ  
                    end  
                end  
                Hist=Hist/sum(Hist);     %�����˳�֮Ϊ��һ��ֱ��ͼ  
                for k=1:256  
                    if Hist(k)~=0  
                       imgn(i,j,num)=imgn(i,j)+Hist(k)*log(1/Hist(k));  %�ֲ���  
                    end  
                end  
            end  
        end  
    end
    imgn = imresize(imgn,[x1,x2]);
%     imshow(imgn,[])  
end

