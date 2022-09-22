#include "kvl_pe.h"
#include "Mappoint.cpp"

/*
    functions ::
        void Erase_BackPoints // Erase Triangulated points that is created behind of the camera direction
*/

#define distance_num_ 2

namespace Keypoint{
    void Erase_BackPoints(vector<Point2f>& points2d, Mat& points3d, Mat Rt, vector<int>& index_of_landmark){
        Mat z_vector = Mat::eye(4, 1, CV_64FC1);
        vector<int> new_index_of_landmark;
        
        z_vector.at<double>(0)=0.0;
        z_vector.at<double>(1)=0.0;
        z_vector.at<double>(2)=1.0;
        z_vector.at<double>(3)=0.0;

        Mat camera_vector=Rt*z_vector;
    
        Mat point3d_homo_vector;

        vector<Point2f> erased_features;
        for(int i=0;i<points3d.rows;i++){
            Mat points3d_vector=Mat::eye(3,1,CV_64FC1);
            points3d_vector.at<double>(0)=points3d.at<double>(i,0)-Rt.at<double>(0,3);
            points3d_vector.at<double>(1)=points3d.at<double>(i,1)-Rt.at<double>(1,3);
            points3d_vector.at<double>(2)=points3d.at<double>(i,2)-Rt.at<double>(2,3);

            //inner product
            double value=(camera_vector.at<double>(0)*points3d_vector.at<double>(0))+
                (camera_vector.at<double>(1)*points3d_vector.at<double>(1))+
                (camera_vector.at<double>(2)*points3d_vector.at<double>(2));
            
            
            if(value<0){
                continue;
            }

            new_index_of_landmark.push_back(index_of_landmark[i]);
            erased_features.push_back
                (cv::Point2f((double)points2d[i].x,(double)points2d[i].y));
            point3d_homo_vector.push_back(points3d.row(i)); 
        }

        index_of_landmark=new_index_of_landmark;
        points3d=point3d_homo_vector.clone();
        points2d=erased_features; 
    }

    //for ORB
    void Erase_BackPoints(vector<Point2f>& points2d, Mat& points3d, Mat Rt, vector<int>& index_of_landmark,vector<cv::KeyPoint>& keypoints){
        Mat z_vector = Mat::eye(4, 1, CV_64FC1);
        vector<int> new_index_of_landmark;
        
        z_vector.at<double>(0)=0.0;
        z_vector.at<double>(1)=0.0;
        z_vector.at<double>(2)=1.0;
        z_vector.at<double>(3)=0.0;

        Mat camera_vector=Rt*z_vector;
    
        Mat point3d_homo_vector;
        vector<cv::KeyPoint> keypoints_temp;

        vector<Point2f> erased_features;
        for(int i=0;i<points3d.rows;i++){
            Mat points3d_vector=Mat::eye(3,1,CV_64FC1);
            points3d_vector.at<double>(0)=points3d.at<double>(i,0)-Rt.at<double>(0,3);
            points3d_vector.at<double>(1)=points3d.at<double>(i,1)-Rt.at<double>(1,3);
            points3d_vector.at<double>(2)=points3d.at<double>(i,2)-Rt.at<double>(2,3);

            //inner product
            double value=(camera_vector.at<double>(0)*points3d_vector.at<double>(0))+
                (camera_vector.at<double>(1)*points3d_vector.at<double>(1))+
                (camera_vector.at<double>(2)*points3d_vector.at<double>(2));
            
            
            if(value<0){
                continue;
            }

            keypoints_temp.push_back(keypoints[i]);//ORB keypoints
            new_index_of_landmark.push_back(index_of_landmark[i]);
            erased_features.push_back
                (cv::Point2f((double)points2d[i].x,(double)points2d[i].y));
            point3d_homo_vector.push_back(points3d.row(i)); 
        }

        keypoints=keypoints_temp;
        index_of_landmark=new_index_of_landmark;
        points3d=point3d_homo_vector.clone();
        points2d=erased_features; 
    }    

    void Erase_BackPoints(vector<Point2f>& points2d, vector<Point2f>& points2d_, Mat& points3d, Mat Rt){
        
        
        Mat z_vector = Mat::eye(4, 1, CV_64FC1);
        
        z_vector.at<double>(0)=0.0;
        z_vector.at<double>(1)=0.0;
        z_vector.at<double>(2)=1.0;
        z_vector.at<double>(3)=0.0;

        Mat camera_vector=Rt*z_vector;
    
        Mat point3d_homo_vector;


        vector<Point2f> erased_features;
        vector<Point2f> erased_features_1;
        printf("Erase backpoint start. size : %d \n",points2d.size());
        for(int i=0;i<points3d.rows;i++){
            Mat points3d_vector=Mat::eye(3,1,CV_64FC1);
            points3d_vector.at<double>(0)=points3d.at<double>(i,0)-Rt.at<double>(0,3);
            points3d_vector.at<double>(1)=points3d.at<double>(i,1)-Rt.at<double>(1,3);
            points3d_vector.at<double>(2)=points3d.at<double>(i,2)-Rt.at<double>(2,3);

            //inner product
            double value=(camera_vector.at<double>(0)*points3d_vector.at<double>(0))+
                (camera_vector.at<double>(1)*points3d_vector.at<double>(1))+
                (camera_vector.at<double>(2)*points3d_vector.at<double>(2));
            
            
            if(value<0){
                continue;
            }

            erased_features.push_back
                (cv::Point2f((double)points2d[i].x,(double)points2d[i].y));
            erased_features_1.push_back
                (cv::Point2f((double)points2d_[i].x,(double)points2d_[i].y));                
            point3d_homo_vector.push_back(points3d.row(i)); 
        }
        printf("Erase backpoint End. size : %d \n",erased_features.size());

        points3d=point3d_homo_vector.clone();
        points2d=erased_features;
        points2d_=erased_features_1;
    }

    void Erase_BackPoints(vector<Point2f>& points2d, vector<Point2f>& points2d_, Mat& points3d, Mat Rt,vector<cv::KeyPoint>& keypoints){
        Mat z_vector = Mat::eye(4, 1, CV_64FC1);
        
        z_vector.at<double>(0)=0.0;
        z_vector.at<double>(1)=0.0;
        z_vector.at<double>(2)=1.0;
        z_vector.at<double>(3)=0.0;

        Mat camera_vector=Rt*z_vector;
        vector<cv::KeyPoint> keypoints_temp;
        Mat point3d_homo_vector;


        vector<Point2f> erased_features;
        vector<Point2f> erased_features_1;
        printf("Erase backpoint start. size : %d \n",points2d.size());
        for(int i=0;i<points3d.rows;i++){
            Mat points3d_vector=Mat::eye(3,1,CV_64FC1);
            points3d_vector.at<double>(0)=points3d.at<double>(i,0)-Rt.at<double>(0,3);
            points3d_vector.at<double>(1)=points3d.at<double>(i,1)-Rt.at<double>(1,3);
            points3d_vector.at<double>(2)=points3d.at<double>(i,2)-Rt.at<double>(2,3);

            //inner product
            double value=(camera_vector.at<double>(0)*points3d_vector.at<double>(0))+
                (camera_vector.at<double>(1)*points3d_vector.at<double>(1))+
                (camera_vector.at<double>(2)*points3d_vector.at<double>(2));
            
            
            if(value<0){
                continue;
            }

            keypoints_temp.push_back(keypoints[i]);
            erased_features.push_back
                (cv::Point2f((double)points2d[i].x,(double)points2d[i].y));
            erased_features_1.push_back
                (cv::Point2f((double)points2d_[i].x,(double)points2d_[i].y));                
            point3d_homo_vector.push_back(points3d.row(i)); 
        }
        printf("Erase backpoint End. size : %d \n",erased_features.size());

        keypoints=keypoints_temp;
        points3d=point3d_homo_vector.clone();
        points2d=erased_features;
        points2d_=erased_features_1;
    }    

    void Erase_Outliers(vector<Point2f>& points2d, Mat& points3d, Mat Rt, Mat Kd, vector<int>& index_of_landmark){
        vector<Point2f> homofeatures;
        vector<int> new_index_of_landmark;
        vector<cv::KeyPoint> keypoints_temp;
        for (int i = 0; i < points3d.rows; i++) {
            if (mode == 1) {
                Mat homopoint = Mat::eye(1, 4, CV_64FC1);
                homopoint.at<double>(0, 3) = 1.0;
                homopoint.at<double>(0, 0) = points3d.at<double>(i, 0);
                homopoint.at<double>(0, 1) = points3d.at<double>(i, 1);
                homopoint.at<double>(0, 2) = points3d.at<double>(i, 2);
                homopoint.at<double>(0, 3) = 1.0;

                transpose(homopoint, homopoint);

                Mat triangulated_points = Kd * Rt * homopoint;
                double temp = triangulated_points.at<double>(2, 0);
                for (int j = 0; j < triangulated_points.rows; j++) {
                    triangulated_points.at<double>(j, 0) /= temp;
                }
                homofeatures.push_back
                (cv::Point2f(triangulated_points.at<double>(0, 0), triangulated_points.at<double>(1, 0)));
            }
        }

        vector<Point2f> tmp_points2d;
        vector<Point3f> tmp_points3d;
        Mat points3d_;

        for(int i=0;i<points3d.rows;i++){
            double x1=homofeatures[i].x;
            double y1=homofeatures[i].y;

            double x2=points2d[i].x;
            double y2=points2d[i].y;

            double distance=sqrt(pow((x1-x2),2)+pow((y1-y2),2));
            if(distance<distance_num_){
                tmp_points2d.push_back(cv::Point2f(points2d[i].x,points2d[i].y));
                tmp_points3d.push_back(cv::Point3f(points3d.at<double>(i,0),points3d.at<double>(i,1),points3d.at<double>(i,2)));
                new_index_of_landmark.push_back(index_of_landmark[i]);
            }
        }

        Mat tmpMat = Mat::eye(tmp_points3d.size(), 3, CV_64FC1);
        for(int i=0;i<tmp_points3d.size();i++){
            tmpMat.at<double>(i, 0) = (double)tmp_points3d[i].x;
            tmpMat.at<double>(i, 1) = (double)tmp_points3d[i].y;
            tmpMat.at<double>(i, 2) = (double)tmp_points3d[i].z;
        }

        points2d=tmp_points2d;
        points3d=tmpMat.clone();
        index_of_landmark=new_index_of_landmark;

    }

    //for ORB
    void Erase_Outliers(vector<Point2f>& points2d, Mat& points3d, Mat Rt, Mat Kd, vector<int>& index_of_landmark, vector<cv::KeyPoint>& keypoints){
        vector<Point2f> homofeatures;
        vector<int> new_index_of_landmark;
        vector<cv::KeyPoint> keypoints_temp;
        for (int i = 0; i < points3d.rows; i++) {
            if (mode == 1) {
                Mat homopoint = Mat::eye(1, 4, CV_64FC1);
                homopoint.at<double>(0, 3) = 1.0;
                homopoint.at<double>(0, 0) = points3d.at<double>(i, 0);
                homopoint.at<double>(0, 1) = points3d.at<double>(i, 1);
                homopoint.at<double>(0, 2) = points3d.at<double>(i, 2);
                homopoint.at<double>(0, 3) = 1.0;

                transpose(homopoint, homopoint);

                Mat triangulated_points = Kd * Rt * homopoint;
                double temp = triangulated_points.at<double>(2, 0);
                for (int j = 0; j < triangulated_points.rows; j++) {
                    triangulated_points.at<double>(j, 0) /= temp;
                }
                homofeatures.push_back
                (cv::Point2f(triangulated_points.at<double>(0, 0), triangulated_points.at<double>(1, 0)));
            }
        }

        vector<Point2f> tmp_points2d;
        vector<Point3f> tmp_points3d;
        Mat points3d_;

        for(int i=0;i<points3d.rows;i++){
            double x1=homofeatures[i].x;
            double y1=homofeatures[i].y;

            double x2=points2d[i].x;
            double y2=points2d[i].y;

            double distance=sqrt(pow((x1-x2),2)+pow((y1-y2),2));
            if(distance<distance_num_){
                keypoints_temp.push_back(keypoints[i]);
                tmp_points2d.push_back(cv::Point2f(points2d[i].x,points2d[i].y));
                tmp_points3d.push_back(cv::Point3f(points3d.at<double>(i,0),points3d.at<double>(i,1),points3d.at<double>(i,2)));
                new_index_of_landmark.push_back(index_of_landmark[i]);
            }
        }

        Mat tmpMat = Mat::eye(tmp_points3d.size(), 3, CV_64FC1);
        for(int i=0;i<tmp_points3d.size();i++){
            tmpMat.at<double>(i, 0) = (double)tmp_points3d[i].x;
            tmpMat.at<double>(i, 1) = (double)tmp_points3d[i].y;
            tmpMat.at<double>(i, 2) = (double)tmp_points3d[i].z;
        }

        keypoints=keypoints_temp;
        points2d=tmp_points2d;
        points3d=tmpMat.clone();
        index_of_landmark=new_index_of_landmark;

    }

    void Erase_Outliers(vector<Point2f>& points2d, Mat& points3d, Mat Rt, Mat Kd){
        vector<Point2f> homofeatures;
        for (int i = 0; i < points3d.rows; i++) {
            if (mode == 1) {
                Mat homopoint = Mat::eye(1, 4, CV_64FC1);
                homopoint.at<double>(0, 3) = 1.0;
                homopoint.at<double>(0, 0) = points3d.at<double>(i, 0);
                homopoint.at<double>(0, 1) = points3d.at<double>(i, 1);
                homopoint.at<double>(0, 2) = points3d.at<double>(i, 2);
                homopoint.at<double>(0, 3) = 1.0;

                transpose(homopoint, homopoint);

                Mat triangulated_points = Kd * Rt * homopoint;
                double temp = triangulated_points.at<double>(2, 0);
                for (int j = 0; j < triangulated_points.rows; j++) {
                    triangulated_points.at<double>(j, 0) /= temp;
                }
                homofeatures.push_back
                (cv::Point2f(triangulated_points.at<double>(0, 0), triangulated_points.at<double>(1, 0)));
            }
        }

        vector<Point2f> tmp_points2d;
        vector<Point3f> tmp_points3d;
        Mat points3d_;

        for(int i=0;i<points3d.rows;i++){
            double x1=homofeatures[i].x;
            double y1=homofeatures[i].y;

            double x2=points2d[i].x;
            double y2=points2d[i].y;

            double distance=sqrt(pow((x1-x2),2)+pow((y1-y2),2));
            if(distance<distance_num_){
                tmp_points2d.push_back(cv::Point2f(points2d[i].x,points2d[i].y));
                tmp_points3d.push_back(cv::Point3f(points3d.at<double>(i,0),points3d.at<double>(i,1),points3d.at<double>(i,2)));
            }
        }

        Mat tmpMat = Mat::eye(tmp_points3d.size(), 3, CV_64FC1);
        for(int i=0;i<tmp_points3d.size();i++){
            tmpMat.at<double>(i, 0) = (double)tmp_points3d[i].x;
            tmpMat.at<double>(i, 1) = (double)tmp_points3d[i].y;
            tmpMat.at<double>(i, 2) = (double)tmp_points3d[i].z;
        }

        points2d=tmp_points2d;
        points3d=tmpMat.clone();

    }

    void Erase_Outliers_1(vector<Point2f>& points2d, vector<Point2f>& points2d1, Mat& points3d, Mat& points3d1, Mat Rt, Mat Kd){
        vector<Point2f> homofeatures;
        for (int i = 0; i < points3d.rows; i++) {
            if (mode == 1) {
                Mat homopoint = Mat::eye(1, 4, CV_64FC1);
                homopoint.at<double>(0, 3) = 1.0;
                homopoint.at<double>(0, 0) = points3d.at<double>(i, 0);
                homopoint.at<double>(0, 1) = points3d.at<double>(i, 1);
                homopoint.at<double>(0, 2) = points3d.at<double>(i, 2);
                homopoint.at<double>(0, 3) = 1.0;

                transpose(homopoint, homopoint);

                Mat triangulated_points = Kd * Rt * homopoint;
                double temp = triangulated_points.at<double>(2, 0);
                for (int j = 0; j < triangulated_points.rows; j++) {
                    triangulated_points.at<double>(j, 0) /= temp;
                }
                homofeatures.push_back
                (cv::Point2f(triangulated_points.at<double>(0, 0), triangulated_points.at<double>(1, 0)));
            }
        }

        vector<Point2f> tmp_points2d;
        vector<Point2f> tmp_points2d_;
        vector<Point3f> tmp_points3d;
        vector<Point3f> tmp_points3d_;
        Mat points3d_;
        bool match=false;
        for(int i=0;i<points3d.rows;i++){
            double x1=homofeatures[i].x;
            double y1=homofeatures[i].y;

            double x2=points2d[i].x;
            double y2=points2d[i].y;

            double distance=sqrt(pow((x1-x2),2)+pow((y1-y2),2));
            if(distance<distance_num_){
                tmp_points2d.push_back(cv::Point2f(points2d[i].x,points2d[i].y));
                tmp_points3d.push_back(cv::Point3f(points3d.at<double>(i,0),points3d.at<double>(i,1),points3d.at<double>(i,2)));
            }
            else{
                tmp_points2d_.push_back(cv::Point2f(points2d[i].x,points2d[i].y));
                tmp_points3d_.push_back(cv::Point3f(points3d.at<double>(i,0),points3d.at<double>(i,1),points3d.at<double>(i,2)));
            }
        }

        Mat tmpMat = Mat::eye(tmp_points3d.size(), 3, CV_64FC1);
        Mat tmpMat1 = Mat::eye(tmp_points3d_.size(), 3, CV_64FC1);

        for(int i=0;i<tmp_points3d.size();i++){
            tmpMat.at<double>(i, 0) = (double)tmp_points3d[i].x;
            tmpMat.at<double>(i, 1) = (double)tmp_points3d[i].y;
            tmpMat.at<double>(i, 2) = (double)tmp_points3d[i].z;
        }

        for(int i=0;i<tmp_points3d_.size();i++){
            tmpMat1.at<double>(i, 0) = (double)tmp_points3d_[i].x;
            tmpMat1.at<double>(i, 1) = (double)tmp_points3d_[i].y;
            tmpMat1.at<double>(i, 2) = (double)tmp_points3d_[i].z;
        }

        points2d=tmp_points2d;
        points3d=tmpMat.clone();

        points2d1=tmp_points2d_;
        points3d=tmpMat1.clone();
    }





    bool Calculate_Angle(Mat& RT, Mat& RT1){
        Mat z_vector = Mat::eye(4, 1, CV_64FC1);
        
        z_vector.at<double>(0)=0.0;
        z_vector.at<double>(1)=0.0;
        z_vector.at<double>(2)=1.0;
        z_vector.at<double>(3)=0.0;

        Mat camera_vector_1=RT*z_vector;
        Mat camera_vector_2=RT1*z_vector;


        double inner=(camera_vector_1.at<double>(0)*camera_vector_2.at<double>(0))+
            (camera_vector_1.at<double>(1)*camera_vector_2.at<double>(1))+
            (camera_vector_1.at<double>(2)*camera_vector_2.at<double>(2));

        double v1=sqrt((camera_vector_1.at<double>(0)*camera_vector_1.at<double>(0))+
            (camera_vector_1.at<double>(1)*camera_vector_1.at<double>(1))+
            (camera_vector_1.at<double>(2)*camera_vector_1.at<double>(2)));

        double v2=sqrt((camera_vector_2.at<double>(0)*camera_vector_2.at<double>(0))+
            (camera_vector_2.at<double>(1)*camera_vector_2.at<double>(1))+
            (camera_vector_2.at<double>(2)*camera_vector_2.at<double>(2)));
            

        double theta=acos(inner/(v1*v2));
        theta=(theta*180)/M_PI;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

        printf("theta is %lf\n",theta);

        if(theta>8){
            return true;
        }else{
            return false;
        }
        
    }

    void Get_ProjMatrix(Mat& point3d, vector<Point2f>& homofeatures, Mat& kd, Mat& Rt){
        for (int i = 0; i < point3d.rows; i++) {
            Mat homopoint = Mat::eye(1, 4, CV_64FC1);
            homopoint.at<double>(0, 0) = point3d.at<double>(i, 0);
            homopoint.at<double>(0, 1) = point3d.at<double>(i, 1);
            homopoint.at<double>(0, 2) = point3d.at<double>(i, 2);
            homopoint.at<double>(0, 3) = 1.0;

            transpose(homopoint, homopoint);

            Mat triangulated_points = kd * Rt * homopoint;
            double temp = triangulated_points.at<double>(2, 0);
            for (int j = 0; j < triangulated_points.rows; j++) {
                triangulated_points.at<double>(j, 0) /= temp;
            }


            homofeatures.push_back
            (cv::Point2f(triangulated_points.at<double>(0, 0), triangulated_points.at<double>(1, 0)));
        }       
    }

    void inlier_extraction(vector<Point2f> currFeatures, vector<Point2d>& currFeatures_2,
                        Mat point3d_world, Mat& point3d_world_2, 
                        vector<int> index_of_landmark, vector<int>& index_of_landmark_2,
                        Mat& Rt1, Mat& Kd){

        Keypoint::Erase_Outliers(currFeatures, point3d_world, Rt1, Kd, index_of_landmark);


        for (int i = 0; i < currFeatures.size(); i++) {
            currFeatures_2.push_back
                (cv::Point2d(currFeatures[i].x,currFeatures[i].y));
        }
        point3d_world_2=point3d_world.clone();
        index_of_landmark_2=index_of_landmark;
    }

    void inlier_extraction_2(vector<Point2f> currFeatures, vector<Point2d>& currFeatures_2,
                        Mat point3d_world, Mat& point3d_world_2, 
                        vector<int> index_of_landmark, vector<int>& index_of_landmark_2,
                        Mat& Rt1, Mat& Kd, int count_keypoint){
        for(int i=0;i<index_of_landmark.size();i++){
            int temp_id=index_of_landmark[i];

            if(temp_id<count_keypoint){
                currFeatures_2.push_back
                    (cv::Point2d(currFeatures[i].x,currFeatures[i].y));
                index_of_landmark_2.push_back(temp_id);
            }
        }
    }


    void Remain_Inlier(vector<Point2f>& points1, Mat& point3d, Mat Rt, vector<int>& index_of_landmark,int number){

        vector<Point2f> temp_points1;
        vector<int> temp_index_of_landmark;
        vector<Point3d> tmp_points3d;

        for(int i=0;i<point3d.rows;i++){
            if(index_of_landmark[i]<number){
                temp_points1.push_back(cv::Point2f(points1[i].x, points1[i].y));
                temp_index_of_landmark.push_back(index_of_landmark[i]);
                tmp_points3d.push_back(cv::Point3d(point3d.at<double>(i,0),point3d.at<double>(i,1),point3d.at<double>(i,2)));
            }
        }
        Mat tmpMat = Mat::eye(tmp_points3d.size(), 3, CV_64FC1);

        for(int i=0;i<tmp_points3d.size();i++){
            tmpMat.at<double>(i, 0) = (double)tmp_points3d[i].x;
            tmpMat.at<double>(i, 1) = (double)tmp_points3d[i].y;
            tmpMat.at<double>(i, 2) = (double)tmp_points3d[i].z;
        }
        points1=temp_points1;
        point3d=tmpMat.clone();
        index_of_landmark=temp_index_of_landmark;

    }

    void ID_matching(Mappoint& land_mark,vector<Point2f>& points1, vector<Point2f>& points2,
        Mat& point3d_1, Mat& point3d_2, vector<int>& index_1, vector<int>& index_2, int distance_num, int keyframeNum){

            for(int i=0;i<point3d_1.rows;i++){
                bool match=false;
                float x2=float(points1[i].x);
                float y2=float(points1[i].y);
                int temp_id=index_1[i];

                for(int j=0;j<point3d_2.rows;j++){
                    float x1=float(points2[i].x);
                    float y1=float(points2[i].y);
                    float distance=sqrt(pow((x1-x2),2)+pow((y1-y2),2));

                    //caculating distance
                    if(distance<distance_num){
                        match=true;
                        index_2[j]=temp_id;

                        //features
                        points2[j].x=points1[i].x;
                        points2[j].y=points1[i].y;

                        //land_mark
                        point3d_2.at<double>(j,0)=point3d_1.at<double>(i,0);
                        point3d_2.at<double>(j,1)=point3d_1.at<double>(i,1);
                        point3d_2.at<double>(j,2)=point3d_1.at<double>(i,2);
                        break;

                    }
                }

            }

    }

    void check_keynum_of_landmark(Mappoint& land_mark,vector<int>& index_1, int num, int keynum){

        for(int i=0;i<index_1.size();i++){
            if(index_1[i]>=num){
                land_mark.push_mappoint_keynum(keynum);
            }
        }
    }
}//end 