#include <iostream>
#include "Anker/readImgDataFile.h"
#include "ImageFeatureTrack/ImageFeatureTracker.hpp"
using namespace std;

int main(int argc,char** argv)
{
  cout << "Hello Feature Tracker!!\n" << endl;
  if(argc < 2)
  {
      printf("Please input config file!\n");
      return -1;
  }
  string settingFile = argv[1];
  FileStorage fsSetting(settingFile,FileStorage::READ);
  if (!fsSetting.isOpened()) {
     printf("File %s is not exist!\n",settingFile.c_str());
     return -1;
  }
  string imgPath = fsSetting["image_path"];
  int claheFlg = fsSetting["clahe"];

  ReadImgDataFile ankerImgReader(imgPath,claheFlg);
  ImageDataType leftImg,rightImg;
  FeatureTracker featureTracker(settingFile);
  while(!ankerImgReader.GetReadOverFlg())
  {
      ankerImgReader.GetStereoFrame(leftImg,rightImg);
      featureTracker.UpdateImage(leftImg.img,rightImg.img);
      //featureTracker.DetectFastCorner();
      featureTracker.PyrLKFeatureTracking();
      //featureTracker.DetectFeature2D();
      cv::waitKey(33);
  }

  return 1;

}
