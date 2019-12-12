#ifndef IMAGEFEATURETRACKER_H
#define IMAGEFEATURETRACKER_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "../Utility/tic_toc.h"

using namespace  std;
using namespace  cv;
class FeatureTracker
{
public:
   FeatureTracker(string _configFile);
  ~FeatureTracker();
   void UpdateImage(Mat& _leftImg,Mat& _rightImg);
   void DetectFeature2D();
   void PyrLKFeatureTracking();
   //fast and gftt exclude descriptor
   void DetectFastCorner();
   void DetectGfttCorner();
   //orb and brisk include descriptor
   void DetectBriskCorner();
   void DetectOrbCorner();
   void SetMask();
   bool GetDetectorConstructFlg(){return detectorSuccessFlg;}
   bool InBorder(const cv::Point2f &pt);
   void ReduceVec(vector<Point2f>& _pointVec,vector<uchar>& _statusVec);
   void ReduceVec(vector<int> &_vec, vector<uchar> &_statusVec);
   void DrawFeature2D(const Mat& _img,const vector<Point2f>& _point2d);
private:
Mat lastLeftImg,lastRightImg;
Mat curLeftImg,curRightImg;
vector<Point2f> lastPoints,curPoints;
vector<int> trackCnt;
vector<int> index;

Ptr<FeatureDetector> detector;
string configFile;
string detectType;
bool detectorSuccessFlg;
int col,row;
int maxCorner,minDist;
Mat mask;
};


#endif
