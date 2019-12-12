#include "readImgDataFile.h"

ReadImgDataFile::ReadImgDataFile(string _path,bool _claheFlg)
{
   path = _path;
   imgList = path + "list.txt";
   claheFlg = _claheFlg;
   printf("[ReadImgFile]:File list.txt path: %s\n",imgList.c_str());
   imgFile.open(imgList.c_str());
   if(!imgFile)
   {
     printf("File %s doesn't exist!\n",imgList.c_str());
     return;
   }
   readOverFlg = false;
}


void ReadImgDataFile::GetStereoFrame(ImageDataType &leftImg, ImageDataType &rightImg)
{

    if(!imgFile.eof())
    {
       readOverFlg = false;
       string img_name;
       string time_now;
       string time_2;
       string delta_time;
       imgFile >> time_now >> img_name >> time_2 >> delta_time;
       double time_s = atof(time_now.c_str()) * 1e-09;
       string stereoName = path + img_name;
       cv::Mat stereoImg = cv::imread(stereoName,cv::IMREAD_GRAYSCALE);
       int cols = stereoImg.cols;
       int rows = stereoImg.rows;
       cv::Rect left_rect(0,0,cols/2,rows);
       cv::Rect right_rect(cols/2,0,cols/2,rows);
       cv::Mat left_img,right_img;
       stereoImg(left_rect).copyTo(left_img);
       stereoImg(right_rect).copyTo(right_img);
       cv::flip(right_img,right_img,0);
       cv::flip(right_img,right_img,1);
       if(claheFlg)
       {
           cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0,cv::Size(8,8));
           clahe->apply(left_img,left_img);
           clahe->apply(right_img,right_img);
           printf("Runing CLAHE to leftImg and rightImg!!!");
       }
       leftImg.time_s = time_s;
       leftImg.img = left_img;
       rightImg.time_s = time_s;
       rightImg.img = right_img;
    }else {
        readOverFlg = true;
    }

}
