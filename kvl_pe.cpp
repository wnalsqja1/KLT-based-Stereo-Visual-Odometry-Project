#include "kvl_pe.h"

#include "viewer.cpp" // visualize Trajectory and feature points using Pangolin and CV 
#include "BundleAdjustment.cpp" // Motion only BA, Local BA, Full BA (Loop Closing)
#include "Mappoint.cpp" 
#include "Keypoint.cpp"
#include "LocalMapping.cpp"
#include "LoopClosing.cpp"
#include "Converter.cpp"

/*viewer parameters*/
#define window_width 1024
#define window_height 768
#define ViewpointX 0
#define ViewpointY -200
#define ViewpointZ -0.1
#define ViewpointF 100

#define local_ba_frame 10
#define distance_num 2

#define CAMERA_MODE 1 // stereo
// #define CAMERA_MODE 2 // mono

extern "C" void G2O_FACTORY_EXPORT g2o_type_VertexSE3(void);

bool loop_closing_switch_2=false;

int main(int argc, char** argv) {
    g2o_type_VertexSE3();

    cout<<"** Visual Odometry of minbum Start. ** "<<endl;

    /*pangolin code*/
    Viewer::my_visualize pangolin_viewer=Viewer::my_visualize(window_width,window_height);
    pangolin_viewer.initialize();
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(window_width, window_height, ViewpointF, ViewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(ViewpointX, ViewpointY, ViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -pangolin_viewer.window_ratio)
                                .SetHandler(new pangolin::Handler3D(s_cam)); 


    //for loop closing ('08.17.)


    vector<Point3d> t_solve_f_vec;
    vector<pair<int, int>> keyframe_number;
    vector<vector<Point2f>> keyframefeatures;
    vector<Mat> keyframe_Rt;

    /*Map point*/
    Mappoint mp;
    Mappoint land_mark;

    int KeyframeNum=0;
    int count_keypoints=0;

    bool draw_=false;

    Mat rvec, tvec; // rotation & translation vector using solvePnP
    Mat point3d_homo, point3d_world; // triangulatePoints output

    Mat Rt0 = Mat::eye(3, 4, CV_64FC1); //prev pose
    Mat Rt1 = Mat::eye(3, 4, CV_64FC1); //next pose
    Mat keyframe_Rt2 = Mat::eye(3, 4, CV_64FC1); //keyframe pose

    Mat R_f, t_f, Rt_f;

    Mat Rt44 = Mat::eye(4, 4, CV_64FC1);//for makeing 4x4 rotation translation matrix in hormogenious
    Mat Rt44_inv = Mat::eye(4, 4, CV_64FC1);//for makeing 4x4 rotation translation matrix in hormogenious

    vector<Point2f> prevFeatures, currFeatures, prevFeatures_2, currFeatures_2;

    vector<cv::Point2f> tempFeatures, v_tempFeatures;
    vector<cv::Point3f> tempFeatures1;

    vector<cv::Point3f> Pango_GTpose, Pango_REpose;

    Mat Pango_Map;
    
    vector<int> index_of_landmark; // store matched index of id -> the order is essential
    vector<int> index_of_landmark_2; //re-detection 

    /*'22.05.31. :: for Local_BA 2 case*/
    vector<int> local_BA_id_list;
    vector<int> local_BA_points_size_list;

    vector<int> local_BA_id_list_inliers;
    vector<int> local_BA_points_size_list_inliers;

    vector<int> loop_BA_id_list;
    vector<int> loop_BA_points_size_list;

    vector<int> loop_BA_id_list_inliers;
    vector<int> loop_BA_points_size_list_inliers;

    Mat key_RT;

    /*'22.07.26. :: for Loop Closing */
    bool Isloopdetected=false;
    int keyframe_prev=0;
    int keyframe_curr=0;


    /*load first features*/
    vector<vector<cv::Mat>> features;
    features.clear();   
    features.reserve(MAX_IMAGE_NUMBER); // memory pre-allocate
    cout<<"load visual vocabulary ... "<<endl;
    OrbVocabulary voc("0630_KITTI00-22_10_4_voc .yml.gz");
    cout<<"done"<<endl;
    cout<<"creating database ..."<<endl;
    OrbDatabase db(voc,false,0); // do not use direct index
    cout<<"done"<<endl;

    LoopClosing::wait();


    /*for ground truth pose visualization*/
    ifstream is;
    double temp_number[12]={0};
    is.open("../kitty/00.txt");
    for(int i=0;i<12;i++){
        is>>temp_number[i];
    }
    Pango_GTpose.push_back(cv::Point3f(temp_number[3],temp_number[7],temp_number[11]));
    
    char filename1[200];
    char filename2[200];


    //for loop closing 
    int loop_clock=0;

// (1) Stereo visual Odometry
#if CAMERA_MODE==1
    vector<vector<cv::KeyPoint>> keypoint_list;

    printf("stereo visual odometry ======\n");
    Mat Kd; Mat Kd1; // intrinsic parameters
    Mat R1 = Mat::eye(3, 3, CV_64FC1);
    Mat R2 = Mat::eye(3, 3, CV_64FC1);
    A.convertTo(Kd, CV_64F);
    A1.convertTo(Kd1, CV_64F);
    cout<<Kd1<<endl;

    vector<Point2f> prevFeatures_L; //left image features
    vector<Point2f> prevFeatures_R; //right image features

    sprintf(filename1, "../kitty/image_0/%06d.png", 0);
    sprintf(filename2, "../kitty/image_1/%06d.png", 0);
    printf("success to load ... kitty image : %d, mode : %d \n", 0, mode);

    //read the first frame image ( Left, right )
    Mat img_1_c = imread(filename1);
    Mat img_2_c = imread(filename2);

    Mat img_1, img_2;
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);    

    Mat mask_temp, descriptors;
    vector<cv::KeyPoint> keypoints;
    
    Ptr<ORB> detector=ORB::create();
    detector->detectAndCompute(img_1, mask_temp, keypoints, descriptors);
    features.push_back(vector<cv::Mat>());

    LoopClosing::changeStructure(descriptors, features.back());
    LoopClosing::testDatabase(features,db,Isloopdetected,keyframe_prev,keyframe_curr,threshold_score, false, KeyframeNum);

    vector<uchar> status;
    Mat mask;


    // 1. initial map create =====================================================
    // 1-1. feature Extraction with Shi-Tomashi Detector - Stereo Image
    featureDetection(img_1, prevFeatures_L);
    // 1-2. feature matching between left image and right image
    featureTracking(img_1, img_2, prevFeatures_L, prevFeatures_R, status);
    printf("prev size : %d, prev right size : %d\n",prevFeatures_L.size(),prevFeatures_R.size());
    // 1-3. feature matching with epipolar constraints = mask
    featureMatching(prevFeatures_L, prevFeatures_R, Rt0, Rt1, mask);
    // 1-4. Triangulation between left features and right features only inliers
    Triangulation(prevFeatures_L, prevFeatures_R, Rt0, Rt1, point3d_world, Kd, Kd1, mask);
    printf("epipolar constraints => 3d points :%d\n",point3d_world.rows);
    // 1-5. erase back points
    Keypoint::Erase_BackPoints(prevFeatures_L, prevFeatures_R, point3d_world, Rt0);            

    //type-casting
    vector<Point2d> prevFeatures_L_double, prevFeatures_R_double;
    VectorTypeCasting_FtoT(prevFeatures_L, prevFeatures_R, prevFeatures_L_double, prevFeatures_R_double);

    // 1-6. insert first 3d_landmark
    // id-generation
    for(int i=0;i<point3d_world.rows;i++){
        index_of_landmark.push_back(i);
    }

    // all points
    local_BA_id_list=index_of_landmark;
    loop_BA_id_list=index_of_landmark;
    land_mark.push_each_keyframe_information(prevFeatures_L_double, 0, index_of_landmark);
    land_mark.InsertMappoint_2(point3d_world, prevFeatures_L_double, 0, index_of_landmark);
    local_BA_points_size_list.push_back(point3d_world.rows);
    loop_BA_points_size_list.push_back(point3d_world.rows);

    // inlier points
    local_BA_id_list_inliers=index_of_landmark;
    loop_BA_id_list_inliers=index_of_landmark;

    land_mark.push_inlier_list(prevFeatures_L_double, 0, index_of_landmark);
    local_BA_points_size_list_inliers.push_back(point3d_world.rows);

    for(int i=0;i<point3d_world.rows;i++){
        land_mark.push_mappoint_keynum(0);
    }

    // keyframe 0 Pose INSERT
    Mat tmp_tvec=Mat::eye(3,1,CV_64FC1);
    Mat tmp_rvec;
    Mat Rt0_tmp=Mat::eye(3,3,CV_64FC1);

    Rt0.rowRange(0,3).colRange(0,3).copyTo(Rt0_tmp.rowRange(0,3).colRange(0,3));
    Rt0.rowRange(0,3).col(3).copyTo(tmp_tvec);
    Rodrigues(Rt0_tmp, tmp_rvec);

    land_mark.InsertPose(tmp_rvec, tmp_tvec);

    // initial map create end =====================================================
    // 2. initial map create ( ORB Part )==========================================

    int ORB_count_keypoints=0;

    Mappoint ORB_land_mark;    
    ORB_land_mark.InsertPose(tmp_rvec,tmp_tvec);

    Mat ORB_Rt0=Rt0.clone(); //left pose
    Mat ORB_Rt1=Rt1.clone(); //right pose

    // for ORB feature matching     
    vector<vector<cv::KeyPoint>> ORB_keypoints_set; // keypoint vector 
    vector<cv::Mat> ORB_descriptors_set; // descriptor vector

    Mat ORB_point3d_world;
    vector<Point2f> ORB_currFeatures;
    vector<Point2d> ORB_currFeatures_double;
    vector<int> ORB_index_of_landmark;

    ORB_keypoints_set.push_back(keypoints);
    ORB_descriptors_set.push_back(descriptors);

    vector<int> ORB_local_BA_id_list;
    vector<int> ORB_local_BA_points_size_list;

    vector<int> ORB_local_BA_id_list_inliers;
    vector<int> ORB_local_BA_points_size_list_inliers;


    // for full BA
    vector<int> ORB_loop_BA_id_list;
    vector<int> ORB_loop_BA_points_size_list;

    vector<int> ORB_loop_BA_id_list_inliers;
    vector<int> ORB_loop_BA_points_size_list_inliers;


    // 2-1. ORB Features matching, ORB_currFeatures, ORB_point3d_world 는 같이 가져가야 한다.
    ORB_featurematching(descriptors, keypoints, img_1, img_2, ORB_Rt0, ORB_Rt1, Kd, Kd1, 
        ORB_currFeatures, ORB_point3d_world, focal, pp);
        // keypoints, ORB_currFeatures, ORB_point3d_world 이거 세개로 control 해야해 

    // // 2-2. generate ID     
    for(int i=0;i<ORB_point3d_world.rows;i++){
        ORB_index_of_landmark.push_back(i);
    }
    // 2-3. outlier curring
    Keypoint::Erase_BackPoints(ORB_currFeatures, ORB_point3d_world, ORB_Rt0, ORB_index_of_landmark, keypoints);
    keypoint_list.push_back(keypoints);
    
    ORB_currFeatures_double=Converter::toPoint2d(ORB_currFeatures);

    ORB_local_BA_id_list=ORB_index_of_landmark;
    ORB_loop_BA_id_list=ORB_index_of_landmark;

    ORB_land_mark.push_each_keyframe_information(ORB_currFeatures_double, 0, ORB_index_of_landmark);
    ORB_land_mark.InsertMappoint_2(ORB_point3d_world, ORB_currFeatures_double, 0, ORB_index_of_landmark);
    ORB_local_BA_points_size_list.push_back(ORB_point3d_world.rows);

    ORB_loop_BA_points_size_list.push_back(ORB_point3d_world.rows);

    // inlier points
    ORB_local_BA_id_list_inliers=ORB_index_of_landmark; // for Local BA
    ORB_loop_BA_id_list_inliers=ORB_index_of_landmark; // for Full BA

    ORB_land_mark.push_inlier_list(ORB_currFeatures_double, 0, ORB_index_of_landmark);
    ORB_local_BA_points_size_list_inliers.push_back(ORB_point3d_world.rows); // for Local BA
    ORB_loop_BA_points_size_list_inliers.push_back(ORB_point3d_world.rows); // for Full BA

    for(int i=0;i<ORB_point3d_world.rows;i++){
        ORB_land_mark.push_mappoint_keynum(0);
    }

    ORB_count_keypoints=ORB_point3d_world.rows;

    // initial map create (ORB part end) ===========================================
    Rt1=Rt0.clone();
    key_RT=Rt0.clone();
    
    KeyframeNum=1;

    prevFeatures=prevFeatures_L;
    Mat prevImage=img_1.clone();
    Mat currImage_l; Mat currImage_r;

    cout<<Rt0<<endl;
    mode=1;
    count_keypoints=point3d_world.rows;

    t_solve_f_vec.push_back(Point3d(0.0,0.0,0.0));


    keyframefeatures.push_back(prevFeatures);
    keyframe_number.push_back(make_pair(0,0));
    keyframe_Rt.push_back(Rt0);



    //frame (2,N)
    for(int numFrame = 1; numFrame < MAX_FRAME; numFrame++){
        // push GT pose into viewer
        for(int i=0; i<12; i++){
            is>>temp_number[i];
        }
        Pango_GTpose.push_back(cv::Point3f(temp_number[3],temp_number[7],temp_number[11]));

        // next frame upload 
        sprintf(filename1, "../kitty/image_0/%06d.png", numFrame);
        sprintf(filename2, "../kitty/image_1/%06d.png", numFrame);

        printf("\n\nsuccess to load ... kitty image : %d, mode : %d \n", numFrame, mode);
        Mat currImage_cl = imread(filename1);
        Mat currImage_cr = imread(filename2);
        cvtColor(currImage_cl, currImage_l, COLOR_BGR2GRAY);
        cvtColor(currImage_cr, currImage_r, COLOR_BGR2GRAY);
        vector<uchar> status1, status2;

        // 2-1. feature tracking using left image
        featureTracking_3(prevImage, currImage_l, prevFeatures, currFeatures, status1, point3d_world, index_of_landmark); //number1 tracking line
        Mat inlier;

        // 2-2. SolvePnPRansac ( Pose Estimation basis left camera pose )
        vector<Point2d> currFeatures_double;
        VectorTypeCasting_FtoT(currFeatures, currFeatures_double);
        solvePnPRansac(point3d_world, currFeatures_double, A, noArray(), rvec, tvec, false, 100, 3.0F, 0.99, inlier, cv::SOLVEPNP_ITERATIVE);
        double inlier_ratio = (double)inlier.rows / (double)point3d_world.rows;
        printf("3d point : %d,  inlier : %d\n", point3d_world.rows, inlier.rows);
        printf("inrier_Ratio : %f\n", inlier_ratio);

        // 2-2-1. Store Inlier points
        vector<cv::Point2d> tempFeatures; vector<cv::Point3d> tempFeatures1;
        vector<int> index_of_landmark_inlier; //inlier id 
        Store_inlier(inlier, currFeatures, point3d_world, tempFeatures, tempFeatures1, index_of_landmark, index_of_landmark_inlier);           
        // 2-3. Motion only BA 
        bundle::motion_only_BA(rvec, tvec, tempFeatures, tempFeatures1, focal, pp);
        cout<<"motion ba done"<<endl;
        UpdateRT(Rt0, Rt1, rvec, tvec, mode);


        Mat final_Rt = Mat::eye(4, 4, CV_64FC1);
        Rt1.copyTo(final_Rt.rowRange(0, 3).colRange(0, 4));
        Mat final_Rt_inv=final_Rt.inv();
        final_Rt_inv.pop_back();

        int rotate_switch=false;
        rotate_switch=Keypoint::Calculate_Angle(Rt1, key_RT);

        Keypoint::Erase_BackPoints(currFeatures, point3d_world, final_Rt_inv, index_of_landmark);


        // 2-4. New keyframe Decision
        if(inlier.rows<150||inlier_ratio<0.7||rotate_switch==true){
            printf("=================redetection is conducted=================\n");
            cout<<"frame number : "<<numFrame<<" keyframe number : "<<KeyframeNum<<endl;
            draw_=true;
            keyframe_number.push_back(make_pair(KeyframeNum,numFrame));

            // 3-0. initial setting
            Mat point3d_world_temp=point3d_world.clone();
            Rt0=Rt1.clone();
            
            // 3-1. New Detection
            prevFeatures_L.clear(); prevFeatures_R.clear(); prevFeatures_L_double.clear(); prevFeatures_R_double.clear();
            featureDetection(currImage_l, prevFeatures_L);
            // 3-2. feature matching between left image and right image
            featureTracking(currImage_l, currImage_r, prevFeatures_L, prevFeatures_R, status);
            // 3-3. feature matching with epipolar constraints = mask, 
            featureMatching(prevFeatures_L, prevFeatures_R, Rt0, Rt1, mask);
            // 3-4. Triangulation between left features and right features only inliers
            Triangulation(prevFeatures_L, prevFeatures_R, Rt0, Rt1, point3d_world, Kd, Kd1, mask);
            // 3-5. erase back points
            Keypoint::Erase_BackPoints(prevFeatures_L, prevFeatures_R, point3d_world, final_Rt_inv);
            VectorTypeCasting_FtoT(prevFeatures_L, prevFeatures_R, prevFeatures_L_double, prevFeatures_R_double);

            Rt1=Rt0.clone();
            // 3-6. ID matching -> currFeatures, point3d_world <-> prevFeatures_L, point3d_world_temp
            vector<int> index_of_landmark_temp;
            for(int i=0;i<point3d_world.rows;i++){
                index_of_landmark_temp.push_back(count_keypoints+i);
            }

            Keypoint::ID_matching(land_mark, currFeatures, prevFeatures_L, point3d_world_temp, point3d_world, index_of_landmark, index_of_landmark_temp, distance_num, KeyframeNum);
            //currFeatures :: before detecting, prevFeatures_L :: after detecting
            //point3d_world_temp :: before detecting, point3d_world :: after detecting
            //index_of_landmark :: before detecting, index_of_landmark_temp :: after detecting
            Keypoint::check_keynum_of_landmark(land_mark,index_of_landmark_temp,count_keypoints,KeyframeNum);

            int temp_count_keypoints=count_keypoints+point3d_world.rows;
            vector<int> temp_index_of_landmark_temp=index_of_landmark_temp;
            vector<Point2f> temp_prevFeatures_L=prevFeatures_L;
            Mat temp_point3d_world=point3d_world.clone();

            //inlier selection
            Keypoint::Erase_Outliers(temp_prevFeatures_L, temp_point3d_world, Rt0, Kd, temp_index_of_landmark_temp);
            index_of_landmark=index_of_landmark_temp;
            vector<Point2d> temp_prevFeatures_L_double;
            VectorTypeCasting_FtoT(temp_prevFeatures_L,temp_prevFeatures_L_double);

            LocalMapping::Insert_LocalList(local_BA_id_list,index_of_landmark,local_BA_points_size_list,point3d_world.rows);
            LocalMapping::Insert_LocalList(local_BA_id_list_inliers, temp_index_of_landmark_temp, local_BA_points_size_list_inliers,temp_point3d_world.rows);

            LocalMapping::Insert_LocalList(loop_BA_id_list,index_of_landmark,loop_BA_points_size_list,point3d_world.rows);
            LocalMapping::Insert_LocalList(loop_BA_id_list_inliers, temp_index_of_landmark_temp, loop_BA_points_size_list_inliers,temp_point3d_world.rows);



            // 3-7) Local BA start
            //insert pose
            if(land_mark.Getrvec_size()<local_ba_frame){
                land_mark.InsertPose(rvec,tvec);
                ORB_land_mark.InsertPose(rvec,tvec);
            }
            else{
                land_mark.pop_Pose();
                ORB_land_mark.pop_Pose();
                land_mark.InsertPose(rvec,tvec);
                ORB_land_mark.InsertPose(rvec,tvec);
            }

            // 3-7. store land_mark and each keyframe keypoitns, inliers and features
            land_mark.push_each_keyframe_information(prevFeatures_L_double, 0, index_of_landmark);
            land_mark.InsertMappoint_2(point3d_world, prevFeatures_L_double, 0, index_of_landmark);
            land_mark.push_inlier_list(temp_prevFeatures_L_double, KeyframeNum, temp_index_of_landmark_temp);

            map<int,Point3d> mappoint=land_mark.getMappoint();
            for(int i=0;i<point3d_world.rows;i++){
                int temp_id=index_of_landmark[i];
                point3d_world.at<double>(i,0)=mappoint[temp_id].x;
                point3d_world.at<double>(i,1)=mappoint[temp_id].y;
                point3d_world.at<double>(i,2)=mappoint[temp_id].z;
            }


            //erase overlab all points
            vector<int> temp_local_BA_id_list, temp_local_BA_id_list_inliers;
            LocalMapping::Erase_OverLab(local_BA_id_list, temp_local_BA_id_list);
            LocalMapping::Erase_OverLab(local_BA_id_list_inliers, temp_local_BA_id_list_inliers);


            if(land_mark.Getrvec_size()>5){
                bundle::Local_BA(land_mark, focal, pp, rvec, tvec, point3d_world, KeyframeNum, index_of_landmark, temp_local_BA_id_list_inliers ,1);                
                UpdateRT(Rt0, Rt1, rvec, tvec, 1);
            }

            LocalMapping::Erase_FirstKeyframe(local_BA_id_list, local_BA_points_size_list, point3d_world.rows, land_mark.Getrvec_size(), local_ba_frame);
            LocalMapping::Erase_FirstKeyframe(local_BA_id_list_inliers, local_BA_points_size_list_inliers, temp_point3d_world.rows,land_mark.Getrvec_size(), local_ba_frame);

            
            detector->detectAndCompute(currImage_l, mask_temp, keypoints, descriptors);

            // 3-- . ORB part =====================================================

            /*
                - ORB Feature matching between left - right image
                - leave only good matches
                - epipolar constraints 

            */
            
            // 3--1. initial setting 
            Mat ORB_final_Rt = Mat::eye(4, 4, CV_64FC1);
            Rt0.copyTo(ORB_final_Rt.rowRange(0, 3).colRange(0, 4));
            Mat ORB_final_Rt_inv=ORB_final_Rt.inv();
            ORB_final_Rt_inv.pop_back();

            ORB_Rt0=Rt0.clone();
            vector<Point2f> ORB_prevFeatures=ORB_currFeatures;
            ORB_currFeatures.clear();

            Mat ORB_point3d_world_temp=ORB_point3d_world.clone();

            ORB_featurematching(descriptors, keypoints, currImage_l, currImage_r, ORB_Rt0, ORB_Rt1, Kd, Kd1, 
                ORB_currFeatures, ORB_point3d_world, focal, pp);

            // 3--2. outlier curring

            vector<Point2f> ORB_currFeatures_temp=ORB_currFeatures;
            Keypoint::Erase_BackPoints(ORB_currFeatures, ORB_currFeatures_temp, ORB_point3d_world, ORB_final_Rt_inv, keypoints);
            ORB_currFeatures_double=Converter::toPoint2d(ORB_currFeatures);

            vector<int> ORB_index_of_landmark_temp;
            // 3--3. generate ID     
            for(int i=0;i<ORB_point3d_world.rows;i++){
                ORB_index_of_landmark_temp.push_back(ORB_count_keypoints+i);
            }
            Keypoint::ID_matching(ORB_land_mark, ORB_prevFeatures, ORB_currFeatures, ORB_point3d_world_temp, ORB_point3d_world, 
                    ORB_index_of_landmark, ORB_index_of_landmark_temp, distance_num, KeyframeNum);
            Keypoint::check_keynum_of_landmark(ORB_land_mark,ORB_index_of_landmark_temp,ORB_count_keypoints,KeyframeNum);


            int ORB_temp_count_keypoints=ORB_count_keypoints+ORB_point3d_world.rows;
            vector<int> ORB_temp_index_of_landmark_temp=ORB_index_of_landmark_temp;
            vector<Point2f> ORB_temp_currFeatures=ORB_currFeatures;
            Mat ORB_temp_point3d_world=ORB_point3d_world.clone();            

            //inlier selection
            Keypoint::Erase_Outliers(ORB_temp_currFeatures, ORB_temp_point3d_world, ORB_Rt0, Kd, ORB_temp_index_of_landmark_temp, keypoints);
            ORB_index_of_landmark=ORB_index_of_landmark_temp;
            vector<Point2d> ORB_temp_currFeatures_double;
            VectorTypeCasting_FtoT(ORB_temp_currFeatures,ORB_temp_currFeatures_double);

            //for local BA
            LocalMapping::Insert_LocalList(ORB_local_BA_id_list, ORB_index_of_landmark, ORB_local_BA_points_size_list, ORB_point3d_world.rows);
            LocalMapping::Insert_LocalList(ORB_local_BA_id_list_inliers, ORB_temp_index_of_landmark_temp, ORB_local_BA_points_size_list_inliers, ORB_temp_point3d_world.rows);

            //for full BA
            LocalMapping::Insert_LocalList(ORB_loop_BA_id_list, ORB_index_of_landmark, ORB_loop_BA_points_size_list, ORB_point3d_world.rows);
            LocalMapping::Insert_LocalList(ORB_loop_BA_id_list_inliers,ORB_temp_index_of_landmark_temp,ORB_loop_BA_points_size_list_inliers, ORB_temp_point3d_world.rows);

            // 3-7. store land_mark and each keyframe keypoitns, inliers and features
            ORB_land_mark.push_each_keyframe_information(ORB_currFeatures_double, 0, ORB_index_of_landmark);
            ORB_land_mark.InsertMappoint_2(ORB_point3d_world, ORB_currFeatures_double, 0, ORB_index_of_landmark);
            ORB_land_mark.push_inlier_list(ORB_temp_currFeatures_double, KeyframeNum, ORB_temp_index_of_landmark_temp);
            keypoint_list.push_back(keypoints);



            map<int,Point3d> ORB_mappoint=ORB_land_mark.getMappoint();
            for(int i=0;i<ORB_point3d_world.rows;i++){
                int temp_id=ORB_index_of_landmark[i];
                ORB_point3d_world.at<double>(i,0)=ORB_mappoint[temp_id].x;
                ORB_point3d_world.at<double>(i,1)=ORB_mappoint[temp_id].y;
                ORB_point3d_world.at<double>(i,2)=ORB_mappoint[temp_id].z;
            }

            //erase overlab all points
            vector<int> ORB_temp_local_BA_id_list, ORB_temp_local_BA_id_list_inliers;
            LocalMapping::Erase_OverLab(ORB_local_BA_id_list, ORB_temp_local_BA_id_list);
            LocalMapping::Erase_OverLab(ORB_local_BA_id_list_inliers, ORB_temp_local_BA_id_list_inliers);
       

            if(land_mark.Getrvec_size()>5){
                bundle::ORB_Local_BA(ORB_land_mark, focal, pp, rvec, tvec, ORB_point3d_world, KeyframeNum, ORB_index_of_landmark, ORB_temp_local_BA_id_list_inliers ,1);                
            }

            LocalMapping::Erase_FirstKeyframe(ORB_local_BA_id_list, ORB_local_BA_points_size_list, ORB_point3d_world.rows, ORB_land_mark.Getrvec_size(), local_ba_frame);
            LocalMapping::Erase_FirstKeyframe(ORB_local_BA_id_list_inliers, ORB_local_BA_points_size_list_inliers, ORB_temp_point3d_world.rows, ORB_land_mark.Getrvec_size(), local_ba_frame);

            // ORB part end ========================================================

            count_keypoints=temp_count_keypoints+1500;
            ORB_count_keypoints=ORB_temp_count_keypoints+1500;
            // setting for next frame
            KeyframeNum++;
            currFeatures=prevFeatures_L;
            Rt1=Rt0.clone();
            key_RT=Rt0.clone();


            keyframefeatures.push_back(currFeatures);
            Mat Rt1_temp=Converter::toCVMat34to44(Rt1);
            Converter::toCVMatInverse(Rt1_temp);
            Converter::toCVMat44to34(Rt1_temp);
            keyframe_Rt.push_back(Rt1_temp);


            if(loop_closing_switch_2==true) loop_clock+=1;
            if(loop_clock>100) loop_closing_switch_2=false;
            /*loop closing*/
            features.push_back(vector<cv::Mat>());
            LoopClosing::changeStructure(descriptors, features.back());
            LoopClosing::testDatabase(features,db,Isloopdetected,keyframe_prev,keyframe_curr,threshold_score,loop_closing_switch_2,KeyframeNum);
            t_solve_f_vec.push_back(Point3d(tvec.at<double>(0),tvec.at<double>(1),tvec.at<double>(2)));


            //if loop detected 
            if(Isloopdetected==1&&loop_closing_switch_2==false){
                //pose graph optimization=====================================
                LoopClosing::wait();
                loop_clock=0;
                loop_closing_switch_2=true; // for loop clock counting (한번 방문했다 는 것을 표시 )

                cout<<"loop closing start"<<"\n";
                cout<<"Isloopdetected: "<<Isloopdetected<<"\n";
                cout<<"keyframe_prev id: "<<keyframe_prev<<"\n";
                cout<<"keyframe_curr id: "<<keyframe_curr<<"\n";        

                // define the optimize optimzier            
                std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linear_solver
                    = g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>>();
                std::unique_ptr<g2o::BlockSolver_6_3> block_solver
                        = g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
                g2o::OptimizationAlgorithm* algorithm
                        = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

                g2o::SparseOptimizer* optimizer=new g2o::SparseOptimizer;
                optimizer->setAlgorithm(algorithm);
                optimizer->setVerbose(true);

                g2o::ParameterSE3Offset* cameraOffset = new g2o::ParameterSE3Offset;
                cameraOffset->setId(0);
                

                //  define the optimizer_sim3
                typedef g2o::BlockSolver<g2o::BlockSolverTraits<7, 7>> BlockSolverType;
                typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>LinearSolverType;
                auto solver = new g2o::OptimizationAlgorithmLevenberg(
                    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

                g2o::SparseOptimizer optimizer_sim3;
                optimizer_sim3.setAlgorithm(solver);
                optimizer_sim3.setVerbose(true);    
                optimizer_sim3.addParameter(cameraOffset);


                vector<g2o::SE3Quat> gt_poses; // store all vertex pose for crating edge 

                // 2-0. point3d setting 
                map<int,Point3d> mappoint_loop=ORB_land_mark.getMappoint();
                int size_prev_vertex_feature=keyframefeatures[keyframe_prev].size();
                vector<pair<int,vector<cv::Point2d>>> each_frame_features=ORB_land_mark.getEach_frame_inliers_features();
                vector<pair<int,vector<int>>> unordered_list_of_keypoints=ORB_land_mark.getUnordered_inliers_list_of_keypoints();

                vector<Point2d> loop_features=each_frame_features[keyframe_prev].second;
                vector<int> loop_index_of_landmark=unordered_list_of_keypoints[keyframe_prev].second;
                int size_prev_vertex_features=loop_features.size();

                //type casting double to float
                vector<Point2f> loop_features_float;
                for(int i=0;i<size_prev_vertex_features;i++){
                    loop_features_float.push_back(cv::Point2f((double)loop_features[i].x,(double)loop_features[i].y));
                }
                
                //create 3d point 
                Mat loop_point3d=Mat::eye(size_prev_vertex_features, 3, CV_64FC1);;
                for(int i=0;i<size_prev_vertex_features;i++){
                    int temp_id=loop_index_of_landmark[i];

                    loop_point3d.at<double>(i,0)=mappoint_loop[temp_id].x;
                    loop_point3d.at<double>(i,1)=mappoint_loop[temp_id].y;
                    loop_point3d.at<double>(i,2)=mappoint_loop[temp_id].z;
                }


                // 2. set vertices into optimizer

                printf("start building Pose Graph ... \n");
                // LoopClosing::setfixedvertices(optimizer, gt_poses);
                LoopClosing::setvariablevertices(optimizer, gt_poses, keyframe_Rt);
                printf("set variable vertices ... \n");
                LoopClosing::setEdgesBetweenVertices(optimizer, gt_poses);
                printf("set Edge between vertices ... \n");
                LoopClosing::setLoopClosing_ORB(optimizer, gt_poses, keyframe_prev, keyframe_curr,
                                                keyframe_number, loop_features_float, loop_point3d, A, keypoint_list);
                // LoopClosing::setLoopClosing(optimizer, gt_poses, keyframe_prev, keyframe_curr,
                //                                 keyframe_number, loop_features_float, loop_point3d, A);                                                

                // // 3. convert SE3 into SIM3
                LoopClosing::setverticesSim3(optimizer, optimizer_sim3, gt_poses);
                LoopClosing::setEdgeSim3(optimizer,optimizer_sim3);

                // // // 3. optimize and update
                optimizer_sim3.initializeOptimization();
                optimizer_sim3.optimize(200);
                Isloopdetected=0;

                Pango_REpose.clear();

                int gt_poses_size=gt_poses.size();


                Mat rvec_temp, tvec_temp;
                cv::Point3d rvec_temp1, tvec_temp1;
                vector<cv::Mat> Relative_Pose;
                keyframe_Rt.clear();

                vector<Point3d> rvec_vec_temp, tvec_vec_temp;
                Mat tmptmp;

                for(int i=0;i<gt_poses_size;i++){
                    //convert vertexsim3expmap to cvmat
                    g2o::VertexSim3Expmap* vtx=static_cast<g2o::VertexSim3Expmap*>(optimizer_sim3.vertex(i));
                    g2o::Sim3 sim3 = vtx->estimate().inverse(); //pose
                    g2o::Sim3 sim3_inv = vtx->estimate(); // projection
                    Eigen::Matrix3d r = sim3_inv.rotation().toRotationMatrix(); // projection
                    Eigen::Vector3d t = sim3_inv.translation();

                    Mat minbum=Converter::toCVMat(r,t); //4x4 cvMatrix Projection 
                    Mat minbum_inv=minbum.inv(); // Pose 
                    
                    //relative pose setting 
                    Mat gt_poses_tmp=Converter::toCVMat(gt_poses[i]); // pose
                    Mat relative_pose=minbum*gt_poses_tmp;// [R1|t1] * [R2|t2]^(-1)
                    relative_pose=relative_pose.inv();
                    relative_pose.pop_back();
                    Relative_Pose.push_back(relative_pose);

                    minbum_inv.pop_back();
                    keyframe_Rt.push_back(minbum_inv);

                    minbum.pop_back();
                    tmptmp=minbum.clone();

                    rvec_temp=Converter::toCVMat(r);
                    tvec_temp=Converter::toCVMat(t);
                    
                    Mat R1_temp;
                    Rodrigues(rvec_temp, R1_temp);

                    rvec_temp1.x=R1_temp.at<double>(0,0);
                    rvec_temp1.y=R1_temp.at<double>(1,0);
                    rvec_temp1.z=R1_temp.at<double>(2,0);

                    tvec_temp1.x=minbum.at<double>(0,3);
                    tvec_temp1.y=minbum.at<double>(1,3);
                    tvec_temp1.z=minbum.at<double>(2,3);

                    land_mark.push_loop_rvec_vec(rvec_temp1);
                    land_mark.push_loop_tvec_vec(tvec_temp1);

                    ORB_land_mark.push_loop_rvec_vec(rvec_temp1);
                    ORB_land_mark.push_loop_tvec_vec(tvec_temp1);                    

                    Pango_REpose.push_back(cv::Point3f(minbum_inv.at<double>(0,3),minbum_inv.at<double>(1,3),minbum_inv.at<double>(2,3)));

                    if(i>=(gt_poses_size-local_ba_frame)){
                        cv::Point3d tmp_rvec=cv::Point3d(rvec_temp1.x,rvec_temp1.y,rvec_temp1.z);
                        cv::Point3d tmp_tvec=cv::Point3d(tvec_temp1.x,tvec_temp1.y,tvec_temp1.z);
                        rvec_vec_temp.push_back(tmp_rvec);
                        tvec_vec_temp.push_back(tmp_tvec);
                    }                 
                }

                land_mark.update_pose(rvec_vec_temp,tvec_vec_temp);
                ORB_land_mark.update_pose(rvec_vec_temp,tvec_vec_temp);

                Mat R1_temp;
                Rodrigues(rvec_temp,R1_temp);
                rvec=R1_temp.clone();
                tvec=tvec_temp.clone();

                Rt1=tmptmp.clone();
                key_RT=tmptmp.clone();                

                // printf("update landmark after PGO start.\n");
                // land_mark.update_landmark_afterPGO(Relative_Pose);
                // printf("update ORB_landmark after PGO start.\n");
                // ORB_land_mark.update_landmark_afterPGO(Relative_Pose);

                printf("update all of landmark finish.\n");

                map<int,Point3d> mappoint1=land_mark.getMappoint();
                for(int i=0;i<point3d_world.rows;i++){
                    int temp_id=index_of_landmark[i];
                    point3d_world.at<double>(i,0)=mappoint1[temp_id].x;
                    point3d_world.at<double>(i,1)=mappoint1[temp_id].y;
                    point3d_world.at<double>(i,2)=mappoint1[temp_id].z;
                }
                
                map<int,Point3d> mappoint2=ORB_land_mark.getMappoint();
                for(int i=0;i<ORB_point3d_world.rows;i++){
                    int temp_id=ORB_index_of_landmark[i];
                    ORB_point3d_world.at<double>(i,0)=mappoint2[temp_id].x;
                    ORB_point3d_world.at<double>(i,1)=mappoint2[temp_id].y;
                    ORB_point3d_world.at<double>(i,2)=mappoint2[temp_id].z;
                }                


                vector<int> temp_loop_BA_id_list, temp_loop_BA_id_list_inliers;
                LocalMapping::Erase_OverLab(loop_BA_id_list, temp_loop_BA_id_list);
                LocalMapping::Erase_OverLab(loop_BA_id_list_inliers, temp_loop_BA_id_list_inliers);

                vector<int> ORB_temp_loop_BA_id_list, ORB_temp_loop_BA_id_list_inliers;
                LocalMapping::Erase_OverLab(ORB_loop_BA_id_list, ORB_temp_loop_BA_id_list);
                LocalMapping::Erase_OverLab(ORB_loop_BA_id_list_inliers, ORB_temp_loop_BA_id_list_inliers);

                bundle::Full_BA(land_mark, focal, pp, rvec, tvec, point3d_world, KeyframeNum-1, index_of_landmark, temp_loop_BA_id_list_inliers ,1);                
                bundle::Full_BA(ORB_land_mark, focal, pp, rvec, tvec, ORB_point3d_world, KeyframeNum-1, ORB_index_of_landmark, ORB_temp_loop_BA_id_list_inliers ,1);                

                //land_mark, ORB_land_mark에 있는, rvec, tvec 현재 기준으로 update ᅙᅢ야 함.

                loop_BA_id_list.clear();
                loop_BA_id_list_inliers.clear();
                loop_BA_points_size_list.clear();
                loop_BA_points_size_list_inliers.clear();

                ORB_loop_BA_id_list.clear();
                ORB_loop_BA_id_list_inliers.clear();
                ORB_loop_BA_points_size_list.clear();
                ORB_loop_BA_points_size_list_inliers.clear();                

                printf("Loop Closing End . . . \n");
            }//Loop Closing End 
        }

        

        
        Pango_Map=point3d_world.clone();
        final_Rt = Mat::eye(4, 4, CV_64FC1);
        Rt1.copyTo(final_Rt.rowRange(0, 3).colRange(0, 4));

        final_Rt_inv=final_Rt.inv();
        final_Rt_inv.pop_back();

        Pango_REpose.push_back   
        (cv::Point3f(final_Rt_inv.at<double>(0,3), final_Rt_inv.at<double>(1,3), final_Rt_inv.at<double>(2,3)));
        Mat prevImage_d = currImage_l.clone();
        cvtColor(prevImage_d, prevImage_d, COLOR_GRAY2BGR);


        prevImage_d=pangolin_viewer.cv_draw_features(prevImage_d, point3d_world, Kd, Rt1, currFeatures);
        imshow("CV viewer", prevImage_d);

        // if (numFrame >= 1) {
        //     waitKey(0);
        // }

        cout<<Rt1<<endl;
        waitKey(1);
        prevImage=currImage_l.clone();
        prevFeatures=currFeatures;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        vector<Point3d> hello=ORB_land_mark.Getvector();
        pangolin_viewer.draw_point(Pango_REpose, Pango_GTpose, hello, Pango_Map);
        pangolin::FinishFrame();
    }





/*
======================================================================================================
Monocular KLT based Visual Odometry
======================================================================================================
*/

// (0) Monocular visual Odometry 
#elif CAMERA_MODE==2
    printf("monocular visual odometry ====== ");
    sprintf(filename1, "../kitty/image_0/%06d.png", 0);
    sprintf(filename2, "../kitty/image_0/%06d.png", 1);

    //read the first two frames from the dataset
    Mat img_1_c = imread(filename1);
    Mat img_2_c = imread(filename2);

    Mat img_1, img_2;
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

    vector<uchar> status;

    featureDetection(img_1, prevFeatures);

    vector<Point2f> keyframeFeatures = prevFeatures;
    vector<Point2f> keyFeatures_removed = prevFeatures;

    featureTracking(img_1, img_2, prevFeatures, currFeatures, status, keyFeatures_removed); //track those features to img_2


    //recovering the pose and the essential matrix
    Mat Kd;
    Mat R1 = Mat::eye(3, 3, CV_64FC1);
    A.convertTo(Kd, CV_64F);
    Mat E, R, t, mask;

    int id_explode=0;

    E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

    Rt0 = Mat::eye(3, 4, CV_64FC1); //prev pose
    Rt1 = Mat::eye(3, 4, CV_64FC1); //next pose

    R.copyTo(Rt1.rowRange(0, 3).colRange(0, 3));
    t.copyTo(Rt1.rowRange(0, 3).col(3));

    Mat final_Rt = Mat::eye(4, 4, CV_64FC1);

    Rt1.copyTo(final_Rt.rowRange(0, 3).colRange(0, 4));
    //cout << final_Rt.inv() << endl;

    Pango_REpose.push_back
    (cv::Point3f(final_Rt.at<double>(0,3), final_Rt.at<double>(1,3), final_Rt.at<double>(2,3)));

    R_f = R.clone();
    t_f = t.clone();

    keyframe_Rt2 = Rt0.clone();
    prevFeatures = currFeatures;

    Mat prevImage = img_2; // I(t)
    Mat currImage; // I(t+1)

    char filename[100];
    for (int numFrame = 2; numFrame < MAX_FRAME; numFrame++) {
        for(int i=0; i<12; i++){
            is>>temp_number[i];
        }

        Pango_GTpose.push_back(cv::Point3f(temp_number[3],temp_number[7],temp_number[11]));

        sprintf(filename, "../kitty/image_0/%06d.png", numFrame);
        printf("\n\nsuccess to load ... kitty image : %d, mode : %d \n", numFrame, mode);
        Mat currImage_c = imread(filename);
        cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
        vector<uchar> status, status1;

        if (mode == 1) {
            if (redetection_switch == true) {//2 :: Re-detection 
                //2-1 re-detection
                redetection_switch = false;
                prevFeatures = prevFeatures_2;
                featureDetection(prevImage, keyframeFeatures);
                keyFeatures_removed = keyframeFeatures;
                prevFeatures_2 = keyframeFeatures;
                keyframe_Rt2 = Rt1.clone();                
            }//2 :: step end 
            

            //3-1 : tracking (1), (2)  
            point3d_world = featureTracking_3(prevImage, currImage, prevFeatures, currFeatures, status, point3d_world, index_of_landmark); //number1 tracking line
            featureTracking(prevImage, currImage, prevFeatures_2, currFeatures_2, status1, keyFeatures_removed); //number2 tracking line
            Mat inlier;

            //3-2 : type-casting and Pose Estimation
            vector<Point2d> currFeatures_double;
            for(int i=0;i<currFeatures.size();i++){
                currFeatures_double.push_back(cv::Point2d(double(currFeatures[i].x),
                double(currFeatures[i].y)));
            }
            solvePnPRansac(point3d_world, currFeatures_double, A, noArray(), rvec, tvec, false, 100, 3.0F, 0.99, inlier, cv::SOLVEPNP_ITERATIVE);

            double inlier_ratio = (double)inlier.rows / (double)point3d_world.rows;
            printf("3d point : %d,  inlier : %d\n", point3d_world.rows, inlier.rows);
            printf("inrier_Ratio : %f\n", inlier_ratio);

            //store inlier points 
            vector<cv::Point2d> tempFeatures;
            vector<cv::Point3d> tempFeatures1;

            vector<int> index_of_landmark_inlier; //inlier id 
            Store_inlier(inlier, currFeatures, point3d_world, tempFeatures, tempFeatures1, index_of_landmark, index_of_landmark_inlier);           
 
            //3-3 : Motion Estimation ( Pose Optimization )
            bundle::motion_only_BA(rvec, tvec, tempFeatures, tempFeatures1, focal, pp);
            cout<<"motion ba done"<<endl;
            UpdateRT(Rt0, Rt1, rvec, tvec, mode);

            Mat final_Rt = Mat::eye(4, 4, CV_64FC1);
            Rt1.copyTo(final_Rt.rowRange(0, 3).colRange(0, 4));
            Mat final_Rt_inv=final_Rt.inv();
            final_Rt_inv.pop_back();

            int rotate_switch=false;
            rotate_switch=Keypoint::Calculate_Angle(Rt1, key_RT);

            //4 : New keyframe selection 
            if (inlier.rows<150||inlier_ratio<0.4||rotate_switch==true) {
                printf("=================redetection is conducted=================\n");
                redetection_switch = true; 
                draw_=true;

                //4-1) Triangulation between keyframe and current frame of re-detection line
                // localBA (0) : Triangulation
                vector<cv::Point2d> triangulation_points1, triangulation_points2;
                for (int i = 0; i < keyFeatures_removed.size(); i++) {
                    triangulation_points1.push_back
                        (cv::Point2d(keyFeatures_removed[i].x,keyFeatures_removed[i].y));
                    triangulation_points2.push_back
                        (cv::Point2d(currFeatures_2[i].x, currFeatures_2[i].y));
                }

                triangulatePoints(Kd * keyframe_Rt2, Kd * Rt1, triangulation_points1, triangulation_points2, point3d_homo);
                point3d_world = convert_14to13(point3d_homo);


                // localBA (1) : Erase outliers
                printf(" point3d : %d\n", point3d_world.rows);
                
                // localBA (2) : id-generation
                for(int i=0;i<currFeatures_2.size();i++){
                    index_of_landmark_2.push_back(count_keypoints+i);
                }

                Keypoint::Erase_BackPoints(currFeatures_2, point3d_world, final_Rt_inv, index_of_landmark_2);
                // Keypoint::Erase_Outliers(currFeatures_2, point3d_world, Rt1, Kd, index_of_landmark_2);

                // localBA (3) : id-matching with inliers 
                for (int i=0;i<tempFeatures.size();i++){
                    bool match=false;
                    float x2=float(tempFeatures[i].x);
                    float y2=float(tempFeatures[i].y);
                    int temp_id=index_of_landmark_inlier[i];

                    for(int j=0;j<currFeatures_2.size();j++){
                        float x1=float(currFeatures_2[j].x);
                        float y1=float(currFeatures_2[j].y);
                        
                        float distance=sqrt(pow((x1-x2),2)+pow((y1-y2),2));
                        if(distance<distance_num){
                            index_of_landmark_2[j]=temp_id;

                            currFeatures_2[j].x=tempFeatures[i].x;
                            currFeatures_2[j].y=tempFeatures[i].y;

                            point3d_world.at<double>(j,0)=tempFeatures1[i].x;
                            point3d_world.at<double>(j,1)=tempFeatures1[i].y;
                            point3d_world.at<double>(j,2)=tempFeatures1[i].z;

                            match=true;
                            break;
                        }
                    }
                }


                int temp_count_keypoints=count_keypoints+index_of_landmark_2.size()+1000;


                vector<int> temp_index_of_landmark_2=index_of_landmark_2;
                vector<Point2f> temp_currFeatures_2_=currFeatures_2;
                Mat temp_point3d_world=point3d_world.clone();

                // Keypoint::Remain_Inlier(temp_currFeatures_2_, temp_point3d_world, Rt1, temp_index_of_landmark_2, count_keypoints);
                Keypoint::Erase_Outliers(temp_currFeatures_2_, temp_point3d_world, Rt1, Kd, temp_index_of_landmark_2);
                printf("remain inlier point3d size : %d \n",temp_point3d_world.rows);

                vector<Point2d> temp_currFeatures_2;
                for(int i=0;i<currFeatures_2.size();i++){ 
                    temp_currFeatures_2.push_back(
                        cv::Point2d(double(temp_currFeatures_2_[i].x), double(temp_currFeatures_2_[i].y)));
                }               

                //type-casting point2f to point2d  ;
                vector<Point2d> currFeatures_2_double;
                for(int i=0;i<currFeatures_2.size();i++){ 
                    currFeatures_2_double.push_back(
                        cv::Point2d(currFeatures_2[i].x,currFeatures_2[i].y));
                }
                // insert information of current keyframe (index, features) 
                land_mark.InsertMappoint_2(point3d_world, currFeatures_2_double, KeyframeNum, index_of_landmark_2);
                land_mark.push_each_keyframe_information(currFeatures_2_double, KeyframeNum, index_of_landmark_2);

                map<int,Point3d> mappoint=land_mark.getMappoint();
                for(int i=0;i<point3d_world.rows;i++){
                    int temp_id=index_of_landmark_2[i];
                        point3d_world.at<double>(i,0)=mappoint[temp_id].x;
                        point3d_world.at<double>(i,1)=mappoint[temp_id].y;
                        point3d_world.at<double>(i,2)=mappoint[temp_id].z;
                }

                index_of_landmark=index_of_landmark_2;
                land_mark.push_inlier_list(temp_currFeatures_2, KeyframeNum, temp_index_of_landmark_2);

                //all
                for(int i=0;i<index_of_landmark.size();i++){
                    local_BA_id_list.push_back(index_of_landmark[i]);
                }
                local_BA_points_size_list.push_back(point3d_world.rows);

                for(int i=0;i<index_of_landmark.size();i++){
                    local_BA_id_list_inliers.push_back(temp_index_of_landmark_2[i]);
                }
                local_BA_points_size_list_inliers.push_back(temp_point3d_world.rows);



                // 3-5) Local BA start
                //insert pose
                if(land_mark.Getrvec_size()<local_ba_frame){
                    land_mark.InsertPose(rvec,tvec);
                }
                else{
                    land_mark.pop_Pose();
                    land_mark.InsertPose(rvec,tvec);
                }
 

                //erase overlab all points
                vector<int> temp_local_BA_id_list=local_BA_id_list;
                sort(temp_local_BA_id_list.begin(),temp_local_BA_id_list.end());
                temp_local_BA_id_list.erase(unique(temp_local_BA_id_list.begin(),temp_local_BA_id_list.end()),temp_local_BA_id_list.end());

                vector<int> temp_local_BA_id_list_inliers=local_BA_id_list_inliers;
                sort(temp_local_BA_id_list_inliers.begin(),temp_local_BA_id_list_inliers.end());
                temp_local_BA_id_list_inliers.erase(unique(temp_local_BA_id_list_inliers.begin(),temp_local_BA_id_list_inliers.end()),temp_local_BA_id_list_inliers.end());

                bundle::Local_BA(land_mark, focal, pp, rvec, tvec, point3d_world, KeyframeNum, index_of_landmark_2, temp_local_BA_id_list_inliers ,1);
                UpdateRT(Rt0, Rt1, rvec, tvec, mode);

                if(land_mark.Getrvec_size()==local_ba_frame){
                    //all points
                    int zahnm=local_BA_points_size_list.front();
                    for(int i=0;i<zahnm;i++){
                        local_BA_id_list.erase(local_BA_id_list.begin());
                    }
                    local_BA_points_size_list.erase(local_BA_points_size_list.begin());
                }

                if(land_mark.Getrvec_size()==local_ba_frame){
                    //all points
                    int zahnm=local_BA_points_size_list_inliers.front();
                    for(int i=0;i<zahnm;i++){
                        local_BA_id_list_inliers.erase(local_BA_id_list_inliers.begin());
                    }
                    local_BA_points_size_list_inliers.erase(local_BA_points_size_list_inliers.begin());
                }
                key_RT=Rt1.clone();
                KeyframeNum++;
                index_of_landmark_2.clear();
                count_keypoints=temp_count_keypoints;
            }
            Pango_Map=point3d_world.clone();

            /* visualization prevFeatures and triangulated Features */
            prevFeatures_2 = currFeatures_2;
            final_Rt = Mat::eye(4, 4, CV_64FC1);
            Rt1.copyTo(final_Rt.rowRange(0, 3).colRange(0, 4));

            final_Rt_inv=final_Rt.inv();
            final_Rt_inv.pop_back();

            Pango_REpose.push_back
            (cv::Point3f(final_Rt_inv.at<double>(0,3), final_Rt_inv.at<double>(1,3), final_Rt_inv.at<double>(2,3)));
            Mat prevImage_d = prevImage.clone();
            cvtColor(prevImage_d, prevImage_d, COLOR_GRAY2BGR);


            if(draw_==false){
                prevImage_d=pangolin_viewer.cv_draw_features(prevImage_d,point3d_world, Kd, Rt1, currFeatures);
            }
            else{
                draw_=false;
                prevImage_d=pangolin_viewer.cv_draw_features(prevImage_d,point3d_world, Kd, Rt1, currFeatures_2);
            }
            imshow("CV viewer", prevImage_d);

            if (numFrame >= 6) {
                waitKey(0);
            }
        }


        // Structure From Motion ( From first frame to 5-frame )====================================================
        else if (mode == 0) { 
            featureTracking(prevImage, currImage, prevFeatures, currFeatures, status, keyFeatures_removed);
            E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
            recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

            UpdateRT(Rt0, Rt1, R, t, mode);

            t_f = t_f + (R_f * t);
            R_f = R * R_f;

            Rt_f = Mat::eye(3, 4, CV_64FC1);
            R_f.copyTo(Rt_f.rowRange(0, 3).colRange(0, 3));
            t_f.copyTo(Rt_f.rowRange(0, 3).col(3));

            Pango_REpose.push_back // for Pose visualization
            (cv::Point3f(Rt_f.at<double>(0,3), Rt_f.at<double>(1,3), Rt_f.at<double>(2,3)));

            if (numFrame == 10) {
                mode = 1;
                redetection_switch = true;
                vector<Point2f> currFeatures_temp;
                vector<Point2f> keyFeatures_removed_temp;

                vector<cv::Point2d> triangulation_points1, triangulation_points2;


                for (int i = 0; i < keyFeatures_removed.size(); i++) {
                    if(mask.at<bool>(i,0)==1){
                        // Point2d type : only triangulation
                        triangulation_points1.push_back
                            (cv::Point2d((double)keyFeatures_removed[i].x, (double)keyFeatures_removed[i].y));
                        triangulation_points2.push_back
                            (cv::Point2d((double)currFeatures[i].x, (double)currFeatures[i].y));

                        // Point2f type : only remaining inliers 
                        currFeatures_temp.push_back
                            (cv::Point2f(currFeatures[i].x,currFeatures[i].y));
                        keyFeatures_removed_temp.push_back
                            (cv::Point2f(keyFeatures_removed[i].x,keyFeatures_removed[i].y));                            
                    }
                }
                /*============================ Triangulation step =================================*/
                Mat Rt_f44 = Mat::eye(4, 4, CV_64FC1);
                Mat Rt_f44_inv;
                Rt_f.copyTo(Rt_f44.rowRange(0, 3).colRange(0, 4));        

                Rt_f44_inv = Rt_f44.inv();
                Mat tepMat1 = Rt_f44_inv.clone();
                tepMat1.pop_back();

                triangulatePoints(Kd * keyframe_Rt2, Kd * tepMat1, triangulation_points1, triangulation_points2, point3d_homo);
                point3d_world = convert_14to13(point3d_homo);

                Keypoint::Erase_BackPoints(currFeatures_temp, keyFeatures_removed_temp, point3d_world, Rt_f);


                currFeatures=currFeatures_temp;
                keyFeatures_removed=keyFeatures_removed_temp;

                //keyframe 1 
                for(int i=0;i<point3d_world.rows;i++){
                    index_of_landmark.push_back(i);
                }

                vector<Point2d> keyFeatures_removed_double;
                vector<Point2d> currFeatures_double;

                for(int i=0;i<currFeatures.size();i++){
                    keyFeatures_removed_double.push_back(
                        cv::Point2d(double(keyFeatures_removed[i].x),double(keyFeatures_removed[i].y)));
                    currFeatures_double.push_back(
                        cv::Point2d(double(currFeatures[i].x),double(currFeatures[i].y)));
                }

                //all points
                land_mark.push_each_keyframe_information(keyFeatures_removed_double, 0, index_of_landmark);
                land_mark.push_each_keyframe_information(currFeatures_double, 1, index_of_landmark);
                land_mark.InsertMappoint_2(point3d_world, currFeatures_double, 1, index_of_landmark);

                local_BA_id_list=index_of_landmark;

                local_BA_points_size_list.push_back(point3d_world.rows);
                local_BA_points_size_list.push_back(point3d_world.rows);

                //inlier points
                land_mark.push_inlier_list(keyFeatures_removed_double, 0, index_of_landmark);
                land_mark.push_inlier_list(currFeatures_double, 1, index_of_landmark);
                local_BA_id_list_inliers=index_of_landmark;

                local_BA_points_size_list_inliers.push_back(point3d_world.rows);
                local_BA_points_size_list_inliers.push_back(point3d_world.rows);

                //keyfrmae 0,1 Pose Insert
                Mat tmp_tvec=Mat::eye(3,1,CV_64FC1);
                Mat tmp_rvec;
                Mat Rt0_tmp=Mat::eye(3,3,CV_64FC1);

                keyframe_Rt2.rowRange(0,3).colRange(0,3).copyTo(Rt0_tmp.rowRange(0,3).colRange(0,3));
                keyframe_Rt2.rowRange(0,3).col(3).copyTo(tmp_tvec);
                Rodrigues(Rt0_tmp, tmp_rvec);

                Mat tmp_tvec1=Mat::eye(3,1,CV_64FC1);
                Mat tmp_rvec1;
                Mat Rt1_tmp=Mat::eye(3,3,CV_64FC1);

                tepMat1.rowRange(0,3).colRange(0,3).copyTo(Rt1_tmp.rowRange(0,3).colRange(0,3));
                tepMat1.rowRange(0,3).col(3).copyTo(tmp_tvec1);
                Rodrigues(Rt1_tmp,tmp_rvec1);

                land_mark.InsertPose(tmp_rvec,tmp_tvec);
                land_mark.InsertPose(tmp_rvec1,tmp_tvec1);

                prevFeatures_2 = currFeatures;
                Rt1 = tepMat1.clone();
                key_RT=tepMat1.clone();
                
                KeyframeNum=2;
                count_keypoints=point3d_world.rows*2+1000;
            }
        }
        if(numFrame<10){
            cout<<Rt_f<<endl;
        }
        else{
            cout<<Rt1<<endl;
        }

        waitKey(1);

        prevImage = currImage.clone();
        prevFeatures = currFeatures;

        /*pangolin code*/
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        vector<Point3d> hello=land_mark.Getvector();
        pangolin_viewer.draw_point(Pango_REpose, Pango_GTpose, hello, Pango_Map);
        pangolin::FinishFrame();
    }
#endif 
    return 0;
}