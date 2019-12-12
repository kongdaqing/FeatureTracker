#ifndef READIMGDATAFILE_H
#define READIMGDATAFILE_H
#include <iostream>
#include <fstream>
#include <queue>
#include <opencv2/opencv.hpp>
using namespace std;
struct ImageDataType
{
    double time_s;
    cv::Mat img;
};

class ReadImgDataFile
{
public:
  ReadImgDataFile(string _path,bool _claheFlg);
  void GetStereoFrame(ImageDataType& leftImg,ImageDataType& rightImg);
  bool GetReadOverFlg(){return readOverFlg;};
private:
  string imgList;
  string path;
  ifstream imgFile;
  bool claheFlg;
  bool readOverFlg;
};

#endif // READIMGDATAFILE_H
