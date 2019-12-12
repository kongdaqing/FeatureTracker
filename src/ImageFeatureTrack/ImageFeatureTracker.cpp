#include "ImageFeatureTracker.hpp"

FeatureTracker::FeatureTracker(string _configFile)
{
    configFile = _configFile;
    FileStorage fsSettings(configFile,FileStorage::READ);
    row = fsSettings["row"];
    col = fsSettings["col"];
    maxCorner = fsSettings["max_corner"];
    minDist = fsSettings["min_dist"];
    string _detectType = fsSettings["detectType"];
    detectType = _detectType;
    printf("[FeatureTracker]:Row is %d Col is %d and max corner is %d\n",row,col,maxCorner);
}

FeatureTracker::~FeatureTracker()
{

}

void FeatureTracker::DetectFeature2D()
{
    if(!detectType.compare("FAST"))
        DetectFastCorner();
    else if (!detectType.compare("ORB")) {
        DetectOrbCorner();
    }else if (!detectType.compare("GFTT")) {
        DetectGfttCorner();
    }else if (!detectType.compare("BRISK")) {
        DetectBriskCorner();
    }
}

void FeatureTracker::UpdateImage(Mat &_leftImg, Mat &_rightImg)
{

    curLeftImg.copyTo(lastLeftImg);
    curLeftImg = _leftImg.clone();
    curRightImg.copyTo(lastRightImg);
    curRightImg = _rightImg.clone();
}

void FeatureTracker::PyrLKFeatureTracking()
{
    static int id = 0;
    curPoints.clear();
    vector<uchar> status;
    vector<float> err;
    if(lastLeftImg.empty())
    {
        goodFeaturesToTrack(curLeftImg,curPoints,maxCorner,0.01,minDist,Mat(),3,false,0.04);
        lastPoints = curPoints;
        for (int i = 0;i < curPoints.size();i++) {
            trackCnt.push_back(1);
            index.push_back(id++);
        }

        return;
    }

    TicToc tic;
    calcOpticalFlowPyrLK(lastLeftImg,curLeftImg,lastPoints,curPoints,status,err,Size(21,21),3);
    printf("[KLTracking]:KLT Tracking cost time is %fms!\n",tic.toc());
    for(int i = 0;i < status.size();i++)
    {
        if(status[i] && !InBorder(curPoints[i]))
            status[i] = 0;
    }
    ReduceVec(curPoints,status);
    ReduceVec(lastPoints,status);
    ReduceVec(trackCnt,status);
    for (auto &p:trackCnt) {
        p++;
    }
    SetMask();
    if(curPoints.size() < maxCorner)
    {
       vector<Point2f> addPoint;
       int addNum = maxCorner - (int)curPoints.size();
       TicToc tic2;
       goodFeaturesToTrack(curLeftImg,addPoint,addNum,0.01,minDist,mask,3,false,0.04);
       printf("[KLTracking]:Add %d points by GFTT cost time is %fms!\n",addNum,tic2.toc());
       for(int i = 0;i < addPoint.size(); i++)
       {
           curPoints.push_back(addPoint[i]);
           trackCnt.push_back(1);
           index.push_back(id++);
       }
    }
    lastPoints = curPoints;
    DrawFeature2D(curLeftImg,curPoints);
}

void FeatureTracker::DetectBriskCorner()
{
    static Ptr<BRISK> brisk = BRISK::create();//Create this type cost much time.
    if(curLeftImg.empty())
        return;

    Mat src = curLeftImg.clone();
    vector<KeyPoint> keypoints;
    TicToc tic;
    brisk->detect(src,keypoints);
    printf("[DetectCorner]: BRISK detect corner cost time %f ms\n",tic.toc());
    drawKeypoints(src,keypoints,src,Scalar(0,255,0),DrawMatchesFlags::DEFAULT);

    imshow("brisk",src);

}

void FeatureTracker::DetectOrbCorner()
{
    static Ptr<ORB> orb = ORB::create(100,2.0f,2,31,0,2,ORB::HARRIS_SCORE,31,20);
    if(curLeftImg.empty())
        return;
    Mat src = curLeftImg.clone();
    vector<KeyPoint> keypoints;
    TicToc tic;
    orb->detect(src,keypoints);

    printf("[DetectCorner]: ORB detect corner cost time %f ms\n",tic.toc());
    drawKeypoints(src,keypoints,src,Scalar(0,255,0),DrawMatchesFlags::DEFAULT);
    imshow("orb",src);
}


void FeatureTracker::DetectFastCorner()
{
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(50,true);
    if(curLeftImg.empty())
        return;
    Mat src = curLeftImg.clone();
    vector<KeyPoint> keypoints;
    TicToc tic;
    fast->detect(src,keypoints,Mat());

    drawKeypoints(src,keypoints,src,Scalar(0,255,0),DrawMatchesFlags::DEFAULT);
    printf("[DetectCorner]: FAST detect corner cost time %f ms\n",tic.toc());
    imshow( "fast", src);
}
/// Parameters for Shi-Tomasi algorithm
void FeatureTracker::DetectGfttCorner()
{
    static Ptr<GFTTDetector> gftt = GFTTDetector::create(100,0.01,10,3,false,0.04);
    if(curLeftImg.empty())
        return;
    Mat src = curLeftImg.clone();
    vector<KeyPoint> keypoints;
    TicToc tic;
    gftt->detect(src,keypoints,Mat());
    printf("[DetectCorner]: GFTT detect corner cost time %f ms\n",tic.toc());
    drawKeypoints(src,keypoints,src,Scalar(0,255,0),DrawMatchesFlags::DEFAULT);
    imshow( "Gftt", src);
}

bool PtsCompare(pair<int,pair<Point2f,int>>& a,pair<int,pair<Point2f,int>>& b)
{
    return a.first > b.first;
}

void FeatureTracker::SetMask()
{
    vector<pair<int,pair<Point2f,int>>> pts;
    for (int i = 0;i < curPoints.size(); i++) {
        pts.push_back(make_pair(trackCnt[i],make_pair(curPoints[i],index[i])));
    }
    sort(pts.begin(),pts.end(),PtsCompare);
    curPoints.clear();
    index.clear();
    trackCnt.clear();
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));
    for(int i = 0; i < pts.size(); i++)
    {
        Point2f pt = pts[i].second.first;
        if(mask.at<uchar>(pt) == 255)
        {
            curPoints.push_back(pt);
            index.push_back(pts[i].second.second);
            trackCnt.push_back(pts[i].first);
            circle(mask,pt,minDist,0,-1);
        }
    }

}


bool FeatureTracker::InBorder(const cv::Point2f &pt)
{
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

void FeatureTracker::ReduceVec(vector<Point2f> &_pointVec, vector<uchar> &_statusVec)
{
    int j = 0;
    for (int i = 0;i < _pointVec.size();i++) {
        if(_statusVec[i] == 1)
            _pointVec[j++] = _pointVec[i];
    }
    _pointVec.resize(j);
}
void FeatureTracker::ReduceVec(vector<int> &_vec, vector<uchar> &_statusVec)
{
    int j = 0;
    for (int i = 0;i < _vec.size();i++) {
        if(_statusVec[i] == 1)
            _vec[j++] = _vec[i];
    }
    _vec.resize(j);
}


void FeatureTracker::DrawFeature2D(const Mat &_img, const vector<Point2f> &_point2d)
{

    Mat colorImg;
    cvtColor(_img,colorImg,CV_GRAY2BGR);
    for(int i=0;i< _point2d.size(); i++)
    {
        circle(colorImg,_point2d[i],4,Scalar(0,255,0),1,8,0);
    }
    imshow("OptKLT",colorImg);
}
