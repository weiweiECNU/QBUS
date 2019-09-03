function [yrt,ydat]=fadat(yd1,yd2,yd3,ytext1,ytext2,ytext3)

ydat=zeros(1,3);
yr1=100*diff(log(yd1(:,7)));yr2=100*diff(log(yd2(:,7)));yr3=100*diff(log(yd3(:,7)));
dy1=datenum(ytext1(2:length(ytext1)));
dy2=datenum(ytext2(2:length(ytext2)));
dy3=datenum(ytext3(2:length(ytext3)));

%match dates
kt=1;
for t=1:min([length(yd1);length(yd2);length(yd3)])
    dy11=yr1(dy3(t)==dy1);dy21=yr2(dy3(t)==dy2);
    if ~(isempty(dy11))&&~(isempty(dy21))
     yrt(kt,:)=[dy11 dy21 yr3(t)];
     ydat(kt)=dy3(t);
     kt=kt+1;
    end
end