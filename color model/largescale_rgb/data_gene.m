clc,clear
close all

% set DAVIS original video path
video_path='E:\czh\data\sci\video dataset\DAVIS1080p';
% set your saving path
save_path='.\';
% load the mask
load('mask.mat')
mask=double(mask);

resolution='Full-Resolution/';
block_size=[1080,1920];
compress_frame=24;

r=[1,0;0,0];g1=[0,1;0,0];g2=[0,0;1,0];b=[0,0;0,1];
rggb=cat(3,r,g1+g2,b);
rgb2raw=repmat(rggb,block_size(1)/2,block_size(2)/2);

gt_save_path=[save_path,'gt/'];
meas_save_path=[save_path,'measurement/'];
if exist(gt_save_path,'dir')==0
   mkdir(gt_save_path);
end
if exist(meas_save_path,'dir')==0
   mkdir(meas_save_path);
end
save(strcat(save_path,'mask.mat'),'mask')

num_yb=1;
name_obj=dir([video_path,'/JPEGImages/',resolution]);
for ii=3:length(name_obj)
   path=[video_path,'/JPEGImages/',resolution,name_obj(ii).name];
   name_frame=dir(path);
   pic1=imread([path,'/',name_frame(3).name]);
   w=size(pic1);
   
   if w(1)<block_size(1)
       continue;
   elseif  w(2)<block_size(2)
       continue;
   end
   
   x=1:block_size(1)/2:w(1)-block_size(1)+1;
   if x(end)<w(1)-block_size(1)
        x=[x,w(1)-block_size(1)];
   end
   
   y=1:block_size(2)/2:w(2)-block_size(2)+1;
   if y(end)<w(2)-block_size(2)
        y=[y,w(2)-block_size(2)];
   end
   
    
   
   for ll=3:compress_frame:length(name_frame)-compress_frame
       
       pic_block=zeros([size(pic1),compress_frame]);
         for mm=1:compress_frame
             pic=imread([path,'/',name_frame(ll+mm-1).name]);
             pic_block(:,:,:,mm)=pic;
         end
         pic_block_mean=mean(pic_block,4);
         d_pic_block=pic_block-pic_block_mean;
         d_pic_block=d_pic_block.^2;
         pic_block_sigma=mean(d_pic_block,4);
         pic_block_sigma=mean(pic_block_sigma,3);
         m=zeros(length(x),length(y));n=1;
         for i=1:length(x)
            for j=1:length(y)
                x1=pic_block_sigma(x(i):x(i)+block_size(1)-1,y(j):y(j)+block_size(2)-1,:);
                a1=max(x1(:))-min(x1(:));
                m(i,j)=a1;
                n=n+1;
            end
         end
         
         [a,index]=sort(m(:),'descend');
        
         for n=1:floor(length(x)*length(y)/4)+1
            x1=index(n); 
            j=floor((x1-1)/length(x))+1;
            i=x1-j*length(x)+length(x);
            
            meas=zeros(block_size(1),block_size(2));
            patch_save=zeros(block_size(1),block_size(2),3,compress_frame);
            for mm=1:compress_frame
                pic=pic_block(x(i):x(i)+block_size(1)-1,y(j):y(j)+block_size(2)-1,:,mm);
                patch_save(:,:,:,mm)=pic;
            end
            save([gt_save_path,num2str(num_yb),'_',name_obj(ii).name,name_frame(ll).name(1:end-4),'_',num2str(n),'.mat'],'patch_save');      

            n1=[num2str(num_yb),'_',name_obj(ii).name,name_frame(ll).name(1:end-4),'_',num2str(n),'.mat'];
            meas=zeros(block_size(1),block_size(2));
            p1=zeros(size(patch_save));
            for iii=1:block_size(2)
                p1(:,iii,:,:)=patch_save(:,block_size(2)+1-iii,:,:);
            end
            for mm=1:compress_frame
                p_1=p1(:,:,:,mm);
            end
            save([gt_save_path,n1(1:length(n1)-4),'_mirror','.mat'],'p1');

            meas=zeros(block_size(1),block_size(2));
            p2=zeros(size(patch_save));
            for iii=1:block_size(1)
                p2(iii,:,:,:)=patch_save(block_size(1)+1-iii,:,:,:);
            end
            for mm=1:compress_frame
                p_2=p2(:,:,:,mm);
            end
            save([gt_save_path,n1(1:length(n1)-4),'_ori90','.mat'],'p2');      

            meas=zeros(block_size(1),block_size(2));
            p3=zeros(size(patch_save));
            for iii=1:block_size(1)
                p3(iii,:,:,:)=p1(block_size(1)+1-iii,:,:,:);
            end
            for mm=1:compress_frame
                p_3=p3(:,:,:,mm);
            end
            save([gt_save_path,n1(1:length(n1)-4),'_mirror90','.mat'],'p3');      
            
            num_yb=num_yb+1;
         end
   end
  
end

