#include "kvl_pe.h"
#include "Converter.cpp"

#define Adjacent_frame 3
#define NIMAGES 4

namespace LoopClosing{
    // ----------------------------------------------------------------------------

    void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
    {
        out.resize(plain.rows);

        for(int i = 0; i < plain.rows; ++i)
        {
            out[i] = plain.row(i);
        }
    }

    // ----------------------------------------------------------------------------

    void testVocCreation(const vector<vector<cv::Mat > > &features)
    {
        // branching factor and depth levels 
        const int k = 10;
        const int L = 4;
        const WeightingType weight = TF_IDF;
        const ScoringType scoring = L1_NORM;

        OrbVocabulary voc(k, L, weight, scoring);

        cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
        voc.create(features);
        cout << "... done!" << endl;

        cout << "Vocabulary information: " << endl
        << voc << endl << endl;

        // lets do something with this vocabulary
        cout << "Matching images against themselves (0 low, 1 high): " << endl;
        BowVector v1, v2;
        for(int i = 0; i < NIMAGES; i++)
        {
            voc.transform(features[i], v1);
            for(int j = 0; j < NIMAGES; j++)
            {
                voc.transform(features[j], v2);
                
                double score = voc.score(v1, v2);
                cout << "Image " << i << " vs Image " << j << ": " << score << endl;
            }
        }

        // save the vocabulary to disk
        cout << endl << "Saving vocabulary..." << endl;
        voc.save("small_voc.yml.gz");
        cout << "Done" << endl;
    }

    // ----------------------------------------------------------------------------

    void testDatabase(const vector<vector<cv::Mat > > &features, OrbDatabase &db
        ,bool &Isloopdetected, int &keyframe_prev_id, int &keyframe_curr_id, double threshold,bool loop_closing_switch_2,int keyframenum)
    {

        db.add(features.back());
        cout << "Database information: " << endl << db << endl;

        // and query the database
        cout << "Querying the database: " << endl;
        // add images to the database
        if(loop_closing_switch_2==true||keyframenum%30==0){
            return;
        }


        int features_size=features.size();
        QueryResults ret;
        for(int i = 0; i < features.size(); i++)
        {
            db.query(features[i], ret, Adjacent_frame);

            // ret[0] is always the same image in this case, because we added it to the 
            // database. ret[1] is the second best match.
            if(i==features_size-1){
                cout << "Searching for Image " << i << ". " << ret << endl;
            }
            if(features_size>Adjacent_frame){
                int entry_id=0;
                double score=0;
                for(int j=0;j<Adjacent_frame;j++){
                    int idid=ret[j].Id;
                    if(abs(i-idid)>Adjacent_frame){
                        entry_id=ret[j].Id;
                        score=ret[j].Score;
                        break;
                    }
                }
                
                if(score!=0){
                    if((score>threshold)&&(i==features_size-1)){
                        cout<<"loop detected"<<"\n";
                        cout<<"Score : "<<score<<"\n";
                        keyframe_prev_id=min(entry_id,i);
                        keyframe_curr_id=max(entry_id,i);
                        Isloopdetected=1;

                        cout <<"prev_keyframe: "<<keyframe_prev_id<<" curr_keyframe : "<<keyframe_curr_id<<endl; 
                    }
                }
            }
        }

        cout << endl;
    }

    void wait()
        {
            cout << endl << "Press enter to continue" << endl;
            getchar();
        }    


                                              
    //build pose graph ( verticle, edge )

    // create new node
    void addPoseVertex(g2o::SparseOptimizer *optimizer, g2o::SE3Quat& pose, bool set_fixed,int id)
    {
        g2o::VertexSE3* v_se3 = new g2o::VertexSE3; // VertexSE3 :: pose vertax variable 
        v_se3->setId(id);
        if(set_fixed) // set_fixed==true :: push initial value ( initial pose ) 
            v_se3->setEstimate(pose);
        v_se3->setFixed(set_fixed); // set_fixed==false ::vertex는 추가 되지만, 초기값이 0인 상태. 
        optimizer->addVertex(v_se3);// add vertex into optimizer 
    }


    // create Edge between pose node1 to pose node2 
    void addEdgePosePose(g2o::SparseOptimizer *optimizer, int id0, int id1, g2o::SE3Quat& relpose)
    {
        g2o::EdgeSE3* edge = new g2o::EdgeSE3; // global pose (SE3Quat) 사이의 relative pose (SE3Quat)  
        edge->setVertex(0, optimizer->vertices().find(id0)->second);
        edge->setVertex(1, optimizer->vertices().find(id1)->second);
        edge->setMeasurement(relpose);
        Eigen::MatrixXd info_matrix = Eigen::MatrixXd::Identity(6,6) * 10.;
        edge->setInformation(info_matrix);
        optimizer->addEdge(edge); // add edge between two vertex
    }

    //converte VertexSE3 to VertexSim3
    void ToVertexSim3(const g2o::VertexSE3 &v_se3,
                    g2o::VertexSim3Expmap *const v_sim3, double scale)
    {
        Eigen::Isometry3d se3 = v_se3.estimate().inverse();
        //vertexSE3 is connected with Isometry 3d
        Eigen::Matrix3d r = se3.rotation(); //se3.rotation -> return rotate
        Eigen::Vector3d t = se3.translation(); //se3.translation -> return translation

        g2o::Sim3 sim3(r, t, 1.0f);

        v_sim3->setEstimate(sim3);
    }

    // Converte EdgeSE3 to EdgeSim3
    void ToEdgeSim3(const g2o::EdgeSE3 &e_se3, g2o::EdgeSim3 *const e_sim3,double scale)
    {
        Eigen::Isometry3d se3 = e_se3.measurement().inverse();
        //EdgeSE3 is connected with Isometry 3d

        Eigen::Matrix3d r = se3.rotation();
        Eigen::Vector3d t = se3.translation();

        g2o::Sim3 sim3(r, t, 1.0f);

        e_sim3->setMeasurement(sim3);
    }

    // Convert Sim3 Vertex to SE3 Vertex
    void ToVertexSE3(const g2o::VertexSim3Expmap& v_sim3,
                    g2o::VertexSE3* const v_se3) {
        g2o::Sim3 sim3 = v_sim3.estimate().inverse();
        Eigen::Matrix3d r = sim3.rotation().toRotationMatrix();
        Eigen::Vector3d t = sim3.translation();
        Eigen::Isometry3d se3;
        se3 = r;
        se3.translation() = t;

        v_se3->setEstimate(se3);
    }

    // build Pose Graph========================================================
    // 1. vertex 
    void setfixedvertices(g2o::SparseOptimizer *optimizer, vector<g2o::SE3Quat>& gt_poses){
        Eigen::Vector3d tran;
        Eigen::Quaterniond quat;
        tran = Eigen::Vector3d(0.0f,0.0f,0.0f);
        quat.setIdentity();
        g2o::SE3Quat pose0(quat, tran);
        addPoseVertex(optimizer, pose0, true, 0);
        gt_poses.push_back(pose0);
    }

    void setvariablevertices(g2o::SparseOptimizer *optimizer, 
                vector<g2o::SE3Quat>& gt_poses, vector<Mat>& keyframe_Rt){

        g2o::SE3Quat pose=Converter::toSE3Quat(keyframe_Rt[0]);
        addPoseVertex(optimizer,pose,true,0);
        gt_poses.push_back(pose);

        for(int i=1;i<keyframe_Rt.size();i++){
            g2o::SE3Quat pose0=Converter::toSE3Quat(keyframe_Rt[i]);
            addPoseVertex(optimizer, pose0, true, i);
            gt_poses.push_back(pose0);         
        }
    }

    void setverticesSim3(g2o::SparseOptimizer *optimizer, g2o::SparseOptimizer &optimizer_sim3,
                vector<g2o::SE3Quat>& gt_poses){

        for(int i=0;i<gt_poses.size();i++){
            g2o::VertexSE3* v_se3 = static_cast<g2o::VertexSE3*>(optimizer->vertex(i));
            g2o::VertexSim3Expmap* v_sim3 = new g2o::VertexSim3Expmap();
            v_sim3->setId(i);
            v_sim3->setMarginalized(false);
            ToVertexSim3(*v_se3, v_sim3, 1.0);
            optimizer_sim3.addVertex(v_sim3);
            if (i == 0) {
                v_sim3->setFixed(true);
            }                     
        }
    }

    // 2. Edge 
    void setEdgesBetweenVertices(g2o::SparseOptimizer *optimizer, vector<g2o::SE3Quat>& gt_poses){
        g2o::SE3Quat relpose;
        // SE3Quat :: 6DOF 3 dimensional pose ( x,y,z qx,qy,qz), it  can initialize from Eigen::Queaterniond, Vector3d
        for(size_t i=1;i<gt_poses.size();i++){
            relpose=gt_poses[i-1].inverse()*gt_poses[i];
            addEdgePosePose(optimizer,i-1,i,relpose);
        }
    }

    void setLoopClosing(g2o::SparseOptimizer *optimizer, vector<g2o::SE3Quat>& gt_poses, 
        int prev_num, int next_num, vector<pair<int,int>> keyframe_num, vector<Point2f> prevFeatures, Mat& point3d_world, Mat A){

        // 1. image load 
        char filename1[200];    
        char filename2[200];    

        sprintf(filename1, "../kitty/image_0/%06d.png", keyframe_num[prev_num].second);
        sprintf(filename2, "../kitty/image_0/%06d.png", keyframe_num[next_num].second);

        Mat img_1_c = imread(filename1);
        Mat img_2_c = imread(filename2);   

        Mat img_1, img_2;
        cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
        cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

        // 2. features matching and calcuilating pose
        vector<Point2f> nextFeatures;
        vector<uchar> status1;
        Mat rvec,tvec,inlier;

        printf("features size : %d, 3d size : %d\n",prevFeatures.size(),point3d_world.rows);
        point3d_world=featureTracking_3(img_1, img_2, prevFeatures, nextFeatures, status1, point3d_world);
        vector<Point2d> currFeatures_double;
        VectorTypeCasting_FtoT(nextFeatures, currFeatures_double);
        printf("features size : %d, 3d size : %d\n",nextFeatures.size(),point3d_world.rows);

        solvePnPRansac(point3d_world, currFeatures_double, A, noArray(), rvec, tvec, false, 100, 3.0F, 0.99, inlier, cv::SOLVEPNP_ITERATIVE);
        Mat Rt0,Rt1;
        UpdateRT(Rt0, Rt1, rvec, tvec, 1);
        Mat Rt1_temp=Converter::toCVMat34to44(Rt1);
        Converter::toCVMatInverse(Rt1_temp);
        Converter::toCVMat44to34(Rt1_temp);       
        // 4. set Edge between loop vertices
        printf("set Edge between loop vertices ...");
        g2o::SE3Quat temp_pose(Converter::toSE3Quat(Rt1_temp));
        g2o::SE3Quat relpose;

        relpose=gt_poses[prev_num].inverse() * temp_pose;
        addEdgePosePose(optimizer, prev_num, next_num, relpose);
    } 

    void setEdgeSim3(g2o::SparseOptimizer *optimizer, g2o::SparseOptimizer &optimizer_sim3){
        int edge_index=0;
        for(auto& tmp : optimizer->edges()){
            g2o::EdgeSE3* e_se3=static_cast<g2o::EdgeSE3*>(tmp);
            int idx0=e_se3->vertex(0)->id();
            int idx1=e_se3->vertex(1)->id();
            g2o::EdgeSim3* e_sim3=new g2o::EdgeSim3();
            ToEdgeSim3(*e_se3,e_sim3,1.0);
            e_sim3->setId(edge_index++);
            e_sim3->setVertex(0, optimizer_sim3.vertices()[idx0]);
            e_sim3->setVertex(1, optimizer_sim3.vertices()[idx1]);
            e_sim3->information() = Eigen::Matrix<double, 7, 7>::Identity();

            optimizer_sim3.addEdge(e_sim3);            
        }
    }

    void setLoopClosing_ORB(g2o::SparseOptimizer *optimizer, vector<g2o::SE3Quat>& gt_poses, 
        int prev_num, int next_num, vector<pair<int,int>> keyframe_num, vector<Point2f>& prevFeatures, Mat& point3d_world, Mat A, vector<vector<cv::KeyPoint>>& keypoint_list){

        //0. initial setting
        vector<cv::KeyPoint> TargetKeypoints, ReferenceKeypoints;
        cv::Mat TargetDescriptor, ReferDescriptor;
        cv::Ptr<cv::DescriptorMatcher> Matcher_ORB=cv::BFMatcher::create(cv::NORM_HAMMING);
        cv::Ptr<cv::Feature2D> orb=cv::ORB::create();

        ReferenceKeypoints=keypoint_list[prev_num];
 
        // 1. image load 
        char filename1[200];    
        char filename2[200];    

        sprintf(filename1, "../kitty/image_0/%06d.png", keyframe_num[prev_num].second);
        sprintf(filename2, "../kitty/image_0/%06d.png", keyframe_num[next_num].second);

        Mat img_1_c = imread(filename1);
        Mat img_2_c = imread(filename2);   

        Mat img_1, img_2;
        cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
        cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

        // 2. features matching and calcuilating pose
        orb->compute(img_1, ReferenceKeypoints, ReferDescriptor); 
        orb->detectAndCompute(img_2, cv::Mat(),TargetKeypoints, TargetDescriptor);// detects keypoints and computes the descriptors
        
        vector<cv::DMatch> matches;
        Matcher_ORB->match(ReferDescriptor, TargetDescriptor, matches);

        // 3. find good matches
        vector<cv::DMatch> good_matches;

        vector<Point2d> points1_ess;
        vector<Point2d> points2_ess;
        
        vector<Point3d> point3d_world_temp;

        //convert keypoint to point2f
        for (int i=0;i<matches.size();i++){
            point3d_world_temp.push_back(cv::Point3d(point3d_world.at<double>(matches[i].queryIdx,0),
                point3d_world.at<double>(matches[i].queryIdx,1),point3d_world.at<double>(matches[i].queryIdx,2)));
            points1_ess.push_back(ReferenceKeypoints[matches[i].queryIdx].pt);
            points2_ess.push_back(TargetKeypoints[matches[i].trainIdx].pt);
        }

        Mat mask;
        Mat E = findEssentialMat(points2_ess, points1_ess, focal, pp, RANSAC, 0.999, 1.0, mask);
        printf("Loop Closing :: starting goodmatches ...%d \n",matches.size());
        // distance 
        double minDist,maxDist;
        minDist=maxDist=matches[0].distance;

        for (int i=1;i<matches.size();i++){
            double dist = matches[i].distance;
            if (dist<minDist) minDist=dist;
            if (dist>maxDist) maxDist=dist;
        }

        vector<DMatch> goodMatches;
        double fTh= 16.0*minDist;
        for (int i=0;i<matches.size();i++){
            if (matches[i].distance <=max(fTh,0.02)){
                
                Point2f pt1 = ReferenceKeypoints[matches[i].queryIdx].pt;
                Point2f pt2 = TargetKeypoints[matches[i].trainIdx].pt;
                float dist_tmp = sqrt ( (pt1.x-pt2.x)*(pt1.x-pt2.x)+(pt1.y-pt2.y)*(pt1.y-pt2.y));

                if(mask.at<bool>(i)==1){
                    goodMatches.push_back(matches[i]);                    
                }
            }
        }
        printf("Loop Closing :: End goodmatches ...%d \n",goodMatches.size());

        points1_ess.clear();
        points2_ess.clear();
        point3d_world_temp.clear();

        for(int i=0;i<goodMatches.size();i++){
            point3d_world_temp.push_back(cv::Point3d(point3d_world.at<double>(goodMatches[i].queryIdx,0),
                point3d_world.at<double>(goodMatches[i].queryIdx,1),point3d_world.at<double>(goodMatches[i].queryIdx,2)));
            points2_ess.push_back(TargetKeypoints[goodMatches[i].trainIdx].pt);
        }
        Mat tmp=Converter::toCVMat(point3d_world_temp);

        // ?. solvePnP Ransacn
        Mat rvec,tvec,inlier;

        solvePnPRansac(tmp, points2_ess, A, noArray(), rvec, tvec, false, 100, 3.0F, 0.99, inlier, cv::SOLVEPNP_ITERATIVE);
        Mat Rt0,Rt1;
        UpdateRT(Rt0, Rt1, rvec, tvec, 1);
        Mat Rt1_temp=Converter::toCVMat34to44(Rt1);
        Converter::toCVMatInverse(Rt1_temp);
        Converter::toCVMat44to34(Rt1_temp);
        // 4. set Edge between loop vertices
        g2o::SE3Quat temp_pose(Converter::toSE3Quat(Rt1_temp));
        g2o::SE3Quat relpose;

        relpose=gt_poses[prev_num].inverse()*temp_pose;
        addEdgePosePose(optimizer,prev_num,next_num,relpose);

        // cv::Mat Result;
        // cv::drawMatches(img_1, ReferenceKeypoints, img_2, TargetKeypoints, goodMatches, Result, 
        //         cv::Scalar::all(-1), cv::Scalar(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        // imshow("Matching Result_ORB", Result);                      
    }

}