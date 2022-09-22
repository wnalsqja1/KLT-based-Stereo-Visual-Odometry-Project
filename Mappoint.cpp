/*
    Mappoint.cpp :: purpose on management of id of keyppoints 
    managing id is useful to implement Bundel Adjustment
*/
#include "kvl_pe.h"
#pragma once

class Mappoint{                                          

    private:
        map<int,Point3d> mappoint; // 3d map points
        map<int,Point2d> featurepoint; // 2d map features
        vector<int> mappoint_keynum;

        vector<pair<int,vector<int>>> list_of_keypoints; // The index of the points that the current frame has

        vector<cv::Mat> keyframe_RT;
        vector<cv::Point3d> rvec_vec;
        vector<cv::Point3d> tvec_vec;

        vector<cv::Point3d> loop_rvec_vec;
        vector<cv::Point3d> loop_tvec_vec;

        vector<pair<int,vector<cv::Point2d>>> each_frame_features; // keyframeNum, feature
        vector<pair<int,vector<cv::Point2d>>> each_frame_inliers_features;
        
        vector<pair<int,vector<int>>> unordered_list_of_keypoints;
        vector<pair<int,vector<int>>> unordered_inliers_list_of_keypoints;

    public:

        void clear();
        void size();

        void InsertPose(Mat RT);
        void InsertPose(Mat& rvec, Mat& tvec);

        void InsertMappoint_1(Mat& point3dworld, vector<Point2d>& currFeatures, int KeyframeNum); // temp function 
        void InsertMappoint_2(Mat& point3dworld, vector<Point2d>& currFeatures, int KeyframeNum, vector<int>& temp_index_of_landmark); // temp function 


        void push_list_of_keypoints(vector<int>& index_of_landmark, int keyframeNum);

        void pop_Pose();
 
        void push_back(int KeyframeNum, cv::Point3d point3dworld);

        void clear_loop_rt();

        void push_loop_rvec_vec(cv::Point3d points1);
        void push_loop_tvec_vec(cv::Point3d points1);


        void push_each_keyframe_information(vector<Point2d> features, int KeyframeNum, vector<int> id_index);

        void push_inlier_list(vector<Point2d>& features, int KeyframeNum, vector<int>&
         inlier_list);

        /*read member variable*/
        int Getrvec_size();
        int Getmap_size();
        int Getloop_rvec_size();

        vector<Point3d> Getvector();
        vector<Point2d> Get2Dvector();

        vector<int> Getmappoint_keynum();
        void push_mappoint_keynum(int num);

        map<int,Point3d> getMappoint();
        map<int,Point2d> getFeaturepoint();
        vector<pair<int,vector<int>>> getList_of_keypoints();
        vector<int> get_curr_List_of_keypoints();
        vector<cv::Mat> getKeyframe_RT();
        
        vector<cv::Point3d> getRvec_vec();
        vector<cv::Point3d> getTvec_vec();

        vector<cv::Point3d> getLoopRvec_vec();
        vector<cv::Point3d> getLoopTvec_vec();

        vector<pair<int,vector<cv::Point2d>>> getEach_frame_feature(); // keyframeNum, feature
        vector<pair<int,vector<cv::Point2d>>> getEach_frame_inliers_features();
        vector<pair<int,vector<int>>> getUnordered_list_of_keypoints();
        vector<pair<int,vector<int>>> getUnordered_inliers_list_of_keypoints();

        void update_land_mark(int id, cv::Point3d points_3d);     
        void update_pose(vector<Point3d> rvec_vec1, vector<Point3d> tvec_vec1);

        int getLast_ID();

        void update_landmark_afterPGO(vector<cv::Mat>& relative_pose);
};

void Mappoint::clear_loop_rt(){
    this->loop_rvec_vec.clear();
    this->loop_tvec_vec.clear();
}


void Mappoint::update_landmark_afterPGO(vector<cv::Mat>& relative_pose){
    int mappoint_size=this->mappoint.size();
    int mappoint_size1=this->mappoint_keynum.size();

    map<int,cv::Point3d>::iterator iter;

    int i=0;
    for(iter=this->mappoint.begin();iter!=this->mappoint.end();iter++){
        cv::Point3d points1;
        cv::Mat cvMat(4,1,CV_64FC1);
        cv::Mat tmp=relative_pose[this->mappoint_keynum[i]].clone();
        cvMat.at<double>(0,0)=iter->second.x;
        cvMat.at<double>(1,0)=iter->second.y;
        cvMat.at<double>(2,0)=iter->second.z;
        cvMat.at<double>(3,0)=1.0f;
        Mat points2=tmp*cvMat;
        points1=cv::Point3d(points2.at<double>(0,0),points2.at<double>(1,0),points2.at<double>(2,0));

        // cout<<"all size : "<<mappoint_size<<" "<<mappoint_size1<<" points1 :: num "<<i<<" points "<<points1<<endl;
        this->update_land_mark(iter->first,points1);

        i++;
    }
}

void Mappoint::clear(){ // member variable initialization
    this->mappoint.clear();
    this->featurepoint.clear();
    this->list_of_keypoints.clear();
    this->keyframe_RT.clear();
}

void Mappoint::InsertPose(Mat RT){
    this->keyframe_RT.push_back(RT);
}

void Mappoint::InsertPose(Mat& rvec, Mat& tvec){
    this->rvec_vec.push_back(cv::Point3d(rvec.at<double>(0),rvec.at<double>(1),rvec.at<double>(2)));
    this->tvec_vec.push_back(cv::Point3d(tvec.at<double>(0),tvec.at<double>(1),tvec.at<double>(2)));
}


//mappoint keynum :: indicates when the ladnmark was created.
vector<int> Mappoint::Getmappoint_keynum(){
    return this->mappoint_keynum;
}
void Mappoint::push_mappoint_keynum(int num){
    this->mappoint_keynum.push_back(num);
}


// Mappoint Inserting 
void Mappoint::InsertMappoint_1(Mat& point3dworld, vector<Point2d>& currFeatures, int keyframeNum){
    /*
        store :: 3d map points
        store :: id-index locations of current keyframe
    */
    vector<int> index_id;

    for(int i=0;i<point3dworld.rows;i++){

        this->mappoint.insert(pair<int,Point3d>(i,
        cv::Point3d(point3dworld.at<double>(i,0), point3dworld.at<double>(i,1), point3dworld.at<double>(i,2))));
    }
}
void Mappoint::InsertMappoint_2(Mat& point3dworld, vector<Point2d>& currFeatures, int KeyframeNum, vector<int>& temp_index_of_landmark){

    for(int i=0;i<point3dworld.rows;i++){
        this->mappoint.insert(pair<int,Point3d>(temp_index_of_landmark[i], 
        cv::Point3d(point3dworld.at<double>(i,0), point3dworld.at<double>(i,1), point3dworld.at<double>(i,2))));
    }
}



void Mappoint::push_loop_rvec_vec(cv::Point3d points1){
    this->loop_rvec_vec.push_back(points1);
}
void Mappoint::push_loop_tvec_vec(cv::Point3d points1){
    this->loop_tvec_vec.push_back(points1);
}


void Mappoint::push_list_of_keypoints(vector<int>& index_of_landmark, int keyframeNum){
    sort(index_of_landmark.begin(),index_of_landmark.end());
    this->list_of_keypoints.push_back(make_pair(keyframeNum, index_of_landmark));
}

void Mappoint::push_back(int KeyframeNum, cv::Point3d point3dworld){
    map<int,Point3d>::reverse_iterator iter1=this->mappoint.rbegin();
    int num=(iter1->first)+1;
    
    this->mappoint.insert(pair<int,Point3d>(num,point3dworld));
}

void Mappoint::push_each_keyframe_information(vector<Point2d> features, int KeyframeNum, vector<int> id_index){
    this->each_frame_features.push_back(make_pair(KeyframeNum, features));
    this->unordered_list_of_keypoints.push_back(make_pair(KeyframeNum, id_index));  
}

void Mappoint::pop_Pose(){
    this->rvec_vec.erase(this->rvec_vec.begin());
    this->tvec_vec.erase(this->tvec_vec.begin());
}



int Mappoint::Getrvec_size(){
    return this->rvec_vec.size();
}
int Mappoint::Getmap_size(){
    int num=this->mappoint.size();
    return num;
}
int Mappoint::Getloop_rvec_size(){
    int num=this->loop_rvec_vec.size();
    return num;
}

//Get only 3d map points in the 'Map'structure as a vector<Point3d> 
vector<Point3d> Mappoint::Getvector(){

    vector<Point3d> mappoints3d;
    int num=0;
    map<int, Point3d>::iterator iter;
    for(iter = mappoint.begin(); iter != mappoint.end(); ++iter){
        mappoints3d.push_back(iter->second);
    }

    return mappoints3d;
}

//Get only 2d Features in the 'Map structure as a vector<Point2d> 
vector<Point2d> Mappoint::Get2Dvector(){

    vector<Point2d> mappoints2d;
    map<int, Point2d>::iterator iter;
    for(iter = featurepoint.begin(); iter != featurepoint.end(); ++iter){
        mappoints2d.push_back(iter->second);
        printf(" key : %d, value : %lf\n",iter->first,iter->second);
    }
    return mappoints2d;
}





/*read variable functions*/
map<int,Point3d> Mappoint::getMappoint(){
    return this->mappoint;
}
map<int,Point2d> Mappoint::getFeaturepoint(){
    return this->featurepoint;
}
vector<pair<int,vector<int>>> Mappoint::getList_of_keypoints(){
    return this->list_of_keypoints;
}
vector<int> Mappoint::get_curr_List_of_keypoints(){
    vector<pair<int,vector<int>>>::iterator iter;
    iter=this->list_of_keypoints.end();
    iter--;
    return iter->second;
}
vector<cv::Mat> Mappoint::getKeyframe_RT(){
    return this->keyframe_RT;
}
vector<cv::Point3d> Mappoint::getRvec_vec(){
    return this->rvec_vec;
}
vector<cv::Point3d> Mappoint::getTvec_vec(){
    return this->tvec_vec;
}
vector<cv::Point3d> Mappoint::getLoopRvec_vec(){
    return this->loop_rvec_vec;
}
vector<cv::Point3d> Mappoint::getLoopTvec_vec(){
    return this->loop_tvec_vec;
}


vector<pair<int,vector<cv::Point2d>>> Mappoint::getEach_frame_feature(){
    return this->each_frame_features;
}
vector<pair<int,vector<cv::Point2d>>> Mappoint::getEach_frame_inliers_features(){
    return this->each_frame_inliers_features;
}

vector<pair<int,vector<int>>> Mappoint::getUnordered_list_of_keypoints(){
    return this->unordered_list_of_keypoints;
}


void Mappoint::update_land_mark(int id, cv::Point3d points_3d){
    this->mappoint[id].x=points_3d.x;
    this->mappoint[id].y=points_3d.y;
    this->mappoint[id].z=points_3d.z;
}

void Mappoint::update_pose(vector<Point3d> rvec_vec1, vector<Point3d> tvec_vec1){
    this->rvec_vec.clear();
    this->tvec_vec.clear();


    this->rvec_vec=rvec_vec1;
    this->tvec_vec=tvec_vec1;
}


/*0608*/

void Mappoint::push_inlier_list(vector<Point2d>& points2d, int KeyframeNum, vector<int>& inlier_list){
    this->unordered_inliers_list_of_keypoints.push_back(make_pair(KeyframeNum,inlier_list));
    this->each_frame_inliers_features.push_back(make_pair(KeyframeNum, points2d));
}

vector<pair<int,vector<int>>> Mappoint::getUnordered_inliers_list_of_keypoints(){
    return this->unordered_inliers_list_of_keypoints;
}

int Mappoint::getLast_ID(){
    map<int,Point3d>::reverse_iterator iter1=this->mappoint.rbegin();
    return iter1->first;
}