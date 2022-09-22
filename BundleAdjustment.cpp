#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include "ceres/ceres.h"

#include "Mappoint.cpp"


#include "ceres/loss_function.h"

/*
    SnavelyReprojectionError :: CostFunctor
    bool operator() (camera, point, residuals)
        camera : 9 parameters
            0-2 parameters -> camera rotation vector (Rodriguez)
            3-5 parameters -> camera translation vector
            6   parameters -> focal length
            7-8 parameters -> radial distortion
        3d point : 3 parameters
        residuals : reprojection error
*/


namespace bundle{
    struct SnavelyReprojectionError
    {
        SnavelyReprojectionError(double observed_x, double observed_y, Eigen::Vector4d point_3d_homo_eig, double focal, double ppx, double ppy)
            : observed_x(observed_x), observed_y(observed_y), point_3d_homo_eig(point_3d_homo_eig), focal(focal), ppx(ppx), ppy(ppy) {}

        template <typename T>
        bool operator()(const T *const rvec_eig,
                        const T *const tvec_eig,
                        T *residuals) const
        {
            // camera[0,1,2] are the angle-axis rotation.

            const T theta = sqrt(rvec_eig[0] * rvec_eig[0] + rvec_eig[1] * rvec_eig[1] + rvec_eig[2] * rvec_eig[2]);

            const T tvec_eig_0 = tvec_eig[0];
            const T tvec_eig_1 = tvec_eig[1];
            const T tvec_eig_2 = tvec_eig[2];

            const T w1 = rvec_eig[0] / theta;
            const T w2 = rvec_eig[1] / theta;
            const T w3 = rvec_eig[2] / theta;

            const T cos = ceres::cos(theta);
            const T sin = ceres::sin(theta);



            Eigen::Matrix<T, 3, 4> Relative_homo_R;
            Relative_homo_R << cos + w1 * w1 * (static_cast<T>(1) - cos), w1 * w2 * (static_cast<T>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<T>(1) - cos) + w2 * sin, tvec_eig_0,
                w1 * w2 * (static_cast<T>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<T>(1) - cos), w2 * w3 * (static_cast<T>(1) - cos) - w1 * sin, tvec_eig_1,
                w1 * w3 * (static_cast<T>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<T>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<T>(1) - cos), tvec_eig_2;

            Eigen::Matrix<T, 3, 1> three_to_p_eig;

            Eigen::Matrix<double, 3, 3> Kd;
            Kd << focal, 0, ppx,
                0, focal, ppy,
                0, 0, 1;

            three_to_p_eig = Kd.cast<T>() * Relative_homo_R * point_3d_homo_eig.cast<T>();

            T predicted_x = (three_to_p_eig[0] / three_to_p_eig[2]);
            T predicted_y = (three_to_p_eig[1] / three_to_p_eig[2]);

            // The error is the difference between the predicted and observed position.
            residuals[0] = predicted_x - T(observed_x);
            residuals[1] = predicted_y - T(observed_y);

            return true;
        }

            double observed_x;
            double observed_y;
            const Eigen::Vector4d point_3d_homo_eig;
            double focal;
            double ppx;
            double ppy;
    };

    void motion_only_BA(cv::Mat& rvec, cv::Mat& tvec, std::vector<cv::Point2d> &points2d, 
                        std::vector<cv::Point3d> &points3d, const double focal, cv::Point2d pp){

        Eigen::Vector3d rvec_eig;
        Eigen::Vector3d tvec_eig;

        rvec_eig[0]=rvec.at<double>(0); rvec_eig[1]=rvec.at<double>(1); rvec_eig[2]=rvec.at<double>(2);
        tvec_eig[0]=tvec.at<double>(0); tvec_eig[1]=tvec.at<double>(1); tvec_eig[2]=tvec.at<double>(2);

        Eigen::MatrixXd points2d_eig(2, points2d.size());
        for(int i=0;i<points2d.size();i++){
            points2d_eig(0,i)=points2d[i].x;
            points2d_eig(1,i)=points2d[i].y;
        }
        Eigen::MatrixXd points3d_eig(4,points3d.size());

        for(int i=0;i<points3d.size();i++){
            points3d_eig(0,i)=points3d[i].x;
            points3d_eig(1,i)=points3d[i].y;
            points3d_eig(2,i)=points3d[i].z;
            points3d_eig(3,i)=1;
        }

        ceres::Problem problem;

        for(int i=0;i<points3d.size();i++){
            ceres::CostFunction* cost_function=
                new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3, 3>
                (new SnavelyReprojectionError(points2d_eig(0,i), points2d_eig(1,i), points3d_eig.col(i), focal, pp.x, pp.y));
            ceres::LossFunction* loss = new ceres::CauchyLoss(1.0);

            problem.AddResidualBlock(cost_function, loss, rvec_eig.data(), tvec_eig.data());

        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 4;
        options.max_num_iterations=100;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        cout<<summary.BriefReport()<<endl;

        for(int i=0;i<3;i++){
            rvec.at<double>(i)=double(rvec_eig[i]);
            tvec.at<double>(i)=double(tvec_eig[i]);
        }
    }



    //======================Local BA Start======================

    struct SnavelyReprojectionError_Local
    {
        SnavelyReprojectionError_Local(double observed_x, double observed_y, double focal, double ppx, double ppy)
            : observed_x(observed_x), observed_y(observed_y), focal(focal), ppx(ppx), ppy(ppy) {}

        template <typename T>
        bool operator()(const T *const rvec_eig,
                        const T *const tvec_eig,
                        const T *const point_3d_homo_eig,
                        T *residuals) const
        {
            // camera[0,1,2] are the angle-axis rotation.

            const T theta = sqrt(rvec_eig[0] * rvec_eig[0] + rvec_eig[1] * rvec_eig[1] + rvec_eig[2] * rvec_eig[2]);

            const T tvec_eig_0 = tvec_eig[0];
            const T tvec_eig_1 = tvec_eig[1];
            const T tvec_eig_2 = tvec_eig[2];

            const T w1 = rvec_eig[0] / theta;
            const T w2 = rvec_eig[1] / theta;
            const T w3 = rvec_eig[2] / theta;

            const T cos = ceres::cos(theta);
            const T sin = ceres::sin(theta);

            Eigen::Matrix<T, 3, 4> Relative_homo_R;
            Relative_homo_R << cos + w1 * w1 * (static_cast<T>(1) - cos), w1 * w2 * (static_cast<T>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<T>(1) - cos) + w2 * sin, tvec_eig_0,
                w1 * w2 * (static_cast<T>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<T>(1) - cos), w2 * w3 * (static_cast<T>(1) - cos) - w1 * sin, tvec_eig_1,
                w1 * w3 * (static_cast<T>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<T>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<T>(1) - cos), tvec_eig_2;

            Eigen::Matrix<T, 1, 3> three_to_p_eig;

            Eigen::Matrix<double, 3, 3> Kd;
            Kd << focal, 0, ppx,
                0, focal, ppy,
                0, 0, 1;


            Eigen::Matrix<T, 4, 1> p3he(point_3d_homo_eig[0], point_3d_homo_eig[1], point_3d_homo_eig[2], static_cast<T>(1));

            three_to_p_eig = Kd.cast<T>() * Relative_homo_R * p3he;

            T predicted_x = (three_to_p_eig[0] / three_to_p_eig[2]);
            T predicted_y = (three_to_p_eig[1] / three_to_p_eig[2]);

            // The error is the difference between the predicted and observed position.
            residuals[0] = predicted_x - T(observed_x);
            residuals[1] = predicted_y - T(observed_y);

            return true;
        }

            double observed_x;
            double observed_y;
            double focal;
            double ppx;
            double ppy;
    };

struct SnavelyReprojectionError_Local_pose_fixed
{
  SnavelyReprojectionError_Local_pose_fixed(double observed_x, double observed_y, double focal, double ppx, double ppy, 
                                            Eigen::VectorXd rvec_eig, Eigen::VectorXd tvec_eig)
      : observed_x(observed_x), observed_y(observed_y), focal(focal), ppx(ppx), ppy(ppy), rvec_eig(rvec_eig), tvec_eig(tvec_eig) {}

  template <typename T>
  bool operator()(const T *const point_3d_homo_eig,
                  T *residuals) const
  {
    // camera[0,1,2] are the angle-axis rotation.

    

    int count = 0;
    const T theta = static_cast<T>( sqrt(rvec_eig[3 * count] * rvec_eig[3 * count] + rvec_eig[3 * count + 1] * rvec_eig[3 * count + 1] + rvec_eig[3 * count + 2] * rvec_eig[3 * count + 2]));

    if(theta==static_cast<T>(0)){

        const T tvec_eig_0 = static_cast<T>(tvec_eig[0]);
        const T tvec_eig_1 = static_cast<T>(tvec_eig[1]);
        const T tvec_eig_2 = static_cast<T>(tvec_eig[2]);


        const T w1 = static_cast<T>(0);
        const T w2 = static_cast<T>(0);
        const T w3 = static_cast<T>(0);

        const T cos = ceres::cos(theta);
        const T sin = ceres::sin(theta);
        Eigen::Matrix<T, 3, 4> Relative_homo_R;
        Relative_homo_R << cos + w1 * w1 * (static_cast<T>(1) - cos), w1 * w2 * (static_cast<T>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<T>(1) - cos) + w2 * sin, tvec_eig_0,
            w1 * w2 * (static_cast<T>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<T>(1) - cos), w2 * w3 * (static_cast<T>(1) - cos) - w1 * sin, tvec_eig_1,
            w1 * w3 * (static_cast<T>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<T>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<T>(1) - cos), tvec_eig_2;

        Eigen::Matrix<T, 1, 3> three_to_p_eig;

        Eigen::Matrix<double, 3, 3> Kd;
        Kd << focal, 0, ppx,
            0, focal, ppy,
            0, 0, 1;
        // Kd = Kd.cast<T>();
        

        Eigen::Matrix<T, 4, 1> p3he(point_3d_homo_eig[0], point_3d_homo_eig[1], point_3d_homo_eig[2], static_cast<T>(1));

        three_to_p_eig = Kd.cast<T>() * Relative_homo_R * p3he;


        T predicted_x = (three_to_p_eig[0] / three_to_p_eig[2]);
        T predicted_y = (three_to_p_eig[1] / three_to_p_eig[2]);

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        // cout<<residuals[0]<<"\n";
        // cout<<residuals[1]<<"\n";
        // waitKey();
        return true;
        
    }
    else{
        const T tvec_eig_0 = static_cast<T>(tvec_eig[3 * count]);
        const T tvec_eig_1 = static_cast<T>(tvec_eig[3 * count + 1]);
        const T tvec_eig_2 = static_cast<T>(tvec_eig[3 * count + 2]);

        const T w1 = static_cast<T>(rvec_eig[3 * count] / theta);
        const T w2 = static_cast<T>(rvec_eig[3 * count + 1] / theta);
        const T w3 = static_cast<T>(rvec_eig[3 * count + 2] / theta);

        const T cos = ceres::cos(theta);
        const T sin = ceres::sin(theta);

        Eigen::Matrix<T, 3, 4> Relative_homo_R;
        Relative_homo_R << cos + w1 * w1 * (static_cast<T>(1) - cos), w1 * w2 * (static_cast<T>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<T>(1) - cos) + w2 * sin, tvec_eig_0,
            w1 * w2 * (static_cast<T>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<T>(1) - cos), w2 * w3 * (static_cast<T>(1) - cos) - w1 * sin, tvec_eig_1,
            w1 * w3 * (static_cast<T>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<T>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<T>(1) - cos), tvec_eig_2;

        Eigen::Matrix<T, 1, 3> three_to_p_eig;

        Eigen::Matrix<double, 3, 3> Kd;
        Kd << focal, 0, ppx,
            0, focal, ppy,
            0, 0, 1;
        // Kd = Kd.cast<T>();
        

        Eigen::Matrix<T, 4, 1> p3he(point_3d_homo_eig[0], point_3d_homo_eig[1], point_3d_homo_eig[2], static_cast<T>(1));

        three_to_p_eig = Kd.cast<T>() * Relative_homo_R * p3he;


        T predicted_x = (three_to_p_eig[0] / three_to_p_eig[2]);
        T predicted_y = (three_to_p_eig[1] / three_to_p_eig[2]);

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        // cout<<residuals[0]<<"\n";
        // cout<<residuals[1]<<"\n";
        // waitKey();
        return true;        
    }
  }



  double observed_x;
  double observed_y;
  double focal;
  double ppx;
  double ppy;
  Eigen::VectorXd rvec_eig;
  Eigen::VectorXd tvec_eig;
};

    // 1. 3D POINTS 는 현재 keyframe의 keypoints
    // 2, R/T 는 이제 2d features에서 그 프레임의 2d features, R/T  넣고 3d points 랑 뭐ㅓ시기 하기

    void Local_BA(Mappoint& land_mark, const double focal, 
                cv::Point2d pp, cv::Mat& rvec, cv::Mat& tvec, 
                cv::Mat& point3d_world, int keyframeNum
                ,vector<int>& index_of_landmark
                ,vector<int>& temp_local_BA_id_list,int count_keypoints){


        map<int,Point3d> mappoint=land_mark.getMappoint(); // 3d map points

        map<int,Point3f>::iterator iter;

        // for(iter=mappoint.begin();iter!=mappoint.end();iter++){
        //     cout<<"Key : "<< iter->first <<"Value : "<<iter->second<<endl;
        // }


        vector<cv::Point3d> rvec_vec=land_mark.getRvec_vec();
        vector<cv::Point3d> tvec_vec=land_mark.getTvec_vec();

        vector<pair<int,vector<cv::Point2d>>> each_frame_features=land_mark.getEach_frame_feature();
        vector<pair<int,vector<cv::Point2d>>> each_frame_inliers_features=land_mark.getEach_frame_inliers_features();
        // vector<pair<int,vector<int>>> unordered_list_of_keypoints=land_mark.getUnordered_list_of_keypoints();
        vector<pair<int,vector<int>>> unordered_list_of_keypoints=land_mark.getUnordered_inliers_list_of_keypoints();

        //size of keyframes 
        int rt_vector_size = rvec_vec.size();
        int points_size=temp_local_BA_id_list.size();
        printf("Local BA :: mappoint size : %d temp_local_BA_size : %d \n",mappoint.size(),points_size);

        Eigen::MatrixXd rvec_eig_local(3,rt_vector_size);//vector of 'rvec' 
        Eigen::MatrixXd tvec_eig_local(3,rt_vector_size);//vector of 'tvec'

        //rotation, translation vector assignments
        for(int i=0;i<rt_vector_size;i++){
            //assign rvec_vec, tvec_Vec to Eigen variables
            rvec_eig_local(0,i)=rvec_vec[i].x;
            rvec_eig_local(1,i)=rvec_vec[i].y;
            rvec_eig_local(2,i)=rvec_vec[i].z;

            tvec_eig_local(0,i)=tvec_vec[i].x;
            tvec_eig_local(1,i)=tvec_vec[i].y;
            tvec_eig_local(2,i)=tvec_vec[i].z;

            printf("Local BA :: %lf %lf %lf, %lf %lf %lf\n",rvec_vec[i].x, rvec_vec[i].y, rvec_vec[i].z, tvec_vec[i].x, tvec_vec[i].y, tvec_vec[i].z);
        }

        Eigen::Vector2d BA_2d_points_eig(2); // 각 프레임의, 2D features Points
        Eigen::MatrixXd BA_3d_points_eig(3, points_size); // 현재 프레임의 3d points

        for(int i=0; i<points_size;i++){
            // assign mappoint to Eigen variables
            int id=temp_local_BA_id_list[i];
            BA_3d_points_eig(0,i)=mappoint[id].x;
            BA_3d_points_eig(1,i)=mappoint[id].y;
            BA_3d_points_eig(2,i)=mappoint[id].z;
            // printf("id : %d    %lf %lf %lf\n",id,BA_3d_points_eig(0,i),BA_3d_points_eig(1,i),BA_3d_points_eig(2,i));
        }
        
        ceres::Problem problem2;
        // unorderes_list_of_keypoints :: keypoints id of each keyframe
        for(int i=0;i<points_size;i++){
            //3d land mark assign(num 1 ~ num N) 
            int temp_id=temp_local_BA_id_list[i]; // id of 3d points 

            //assign 2d_features_points of each keyframes
            for(int j=0;j<rt_vector_size;j++){
                int curr_keyframe=(keyframeNum-rt_vector_size+j+1);

                int a=0;
                bool match=false;
                int k_index=0;
                int temp_size=unordered_list_of_keypoints.at(curr_keyframe).second.size();

                for(int k=0;k<temp_size;k++){
                    a=unordered_list_of_keypoints.at(curr_keyframe).second[k];

                    if(temp_id==a){
                        match=true;
                        k_index=k;
                        break;
                    }
                }

                if(match==false){
                    continue;
                }

                else{
                    BA_2d_points_eig[0]=each_frame_inliers_features[curr_keyframe].second.at(k_index).x;
                    BA_2d_points_eig[1]=each_frame_inliers_features[curr_keyframe].second.at(k_index).y;

                    if(j>5){
                        ceres::CostFunction* cost_function2 = 
                        new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local, 2, 3, 3, 3>(
                            new SnavelyReprojectionError_Local(BA_2d_points_eig[0], BA_2d_points_eig[1], focal, pp.x, pp.y)
                        );
                        
                        ceres::LossFunction* loss = new ceres::CauchyLoss(1.0);


                        problem2.AddResidualBlock(cost_function2,
                                                loss,
                                                rvec_eig_local.col(j).data(),
                                                tvec_eig_local.col(j).data(),
                                                BA_3d_points_eig.col(i).data());   
                    }

                    else{
                        Eigen::Vector3d rvec_eig;
                        Eigen::Vector3d tvec_eig;

                        rvec_eig[0]=rvec_vec[j].x; 
                        rvec_eig[1]=rvec_vec[j].y; 
                        rvec_eig[2]=rvec_vec[j].z;

                        tvec_eig[0]=tvec_vec[j].x; 
                        tvec_eig[1]=tvec_vec[j].y; 
                        tvec_eig[2]=tvec_vec[j].z;

                        ceres::CostFunction* cost_function3 = 
                        new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local_pose_fixed, 2, 3>(
                            new SnavelyReprojectionError_Local_pose_fixed(BA_2d_points_eig[0],BA_2d_points_eig[1],focal,pp.x,pp.y,rvec_eig,tvec_eig)
                        );

                        ceres::LossFunction* loss = new ceres::CauchyLoss(1.0);

                        problem2.AddResidualBlock(cost_function3,
                                                loss,
                                                BA_3d_points_eig.col(i).data());                     
                    }
                }            
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = 4;
        options.max_num_iterations=100;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem2, &summary);

        cout<<summary.BriefReport()<<endl;
        cout<<"Local BA :: local BA done\n"<<endl;

        for(int i=0;i<rt_vector_size;i++){
            double rvec_eig_1=rvec_eig_local(0,i);
            double rvec_eig_2=rvec_eig_local(1,i);
            double rvec_eig_3=rvec_eig_local(2,i);

            double tvec_eig_1=tvec_eig_local(0,i);
            double tvec_eig_2=tvec_eig_local(1,i);
            double tvec_eig_3=tvec_eig_local(2,i); 

            rvec_vec[i].x=rvec_eig_1;
            rvec_vec[i].y=rvec_eig_2;
            rvec_vec[i].z=rvec_eig_3;

            tvec_vec[i].x=tvec_eig_1;
            tvec_vec[i].y=tvec_eig_2;
            tvec_vec[i].z=tvec_eig_3;
        }
        land_mark.update_pose(rvec_vec,tvec_vec);


        int b=rt_vector_size-1;
        rvec.at<double>(0)=rvec_vec.at(b).x;
        rvec.at<double>(1)=rvec_vec.at(b).y;
        rvec.at<double>(2)=rvec_vec.at(b).z;

        tvec.at<double>(0)=tvec_vec.at(b).x;
        tvec.at<double>(1)=tvec_vec.at(b).y;
        tvec.at<double>(2)=tvec_vec.at(b).z;

        for(int i=0;i<points_size;i++){
            // land_mark (map structure mappoints update )
            int temp_id=temp_local_BA_id_list[i];

            double points3d_1=BA_3d_points_eig(0,i);
            double points3d_2=BA_3d_points_eig(1,i);
            double points3d_3=BA_3d_points_eig(2,i);

            mappoint[temp_id].x=points3d_1;
            mappoint[temp_id].y=points3d_2;
            mappoint[temp_id].z=points3d_3;

            cv::Point3d points_3d=cv::Point3d(points3d_1,points3d_2,points3d_3);
            land_mark.update_land_mark(temp_id, points_3d);
        }

        //points3d (current 3d points update)

        for(int i=0;i<point3d_world.rows;i++){
            int temp_id=index_of_landmark[i];
                point3d_world.at<double>(i,0)=mappoint[temp_id].x;
                point3d_world.at<double>(i,1)=mappoint[temp_id].y;
                point3d_world.at<double>(i,2)=mappoint[temp_id].z;
        }
    }



// for ORB _ local BA (motion fix)
    void ORB_Local_BA(Mappoint& land_mark, const double focal, 
                cv::Point2d pp, cv::Mat& rvec, cv::Mat& tvec, 
                cv::Mat& point3d_world, int keyframeNum
                ,vector<int>& index_of_landmark
                ,vector<int>& temp_local_BA_id_list,int count_keypoints){

        map<int,Point3d> mappoint=land_mark.getMappoint(); // 3d map points

        map<int,Point3f>::iterator iter;

        // for(iter=mappoint.begin();iter!=mappoint.end();iter++){
        //     cout<<"Key : "<< iter->first <<"Value : "<<iter->second<<endl;
        // }

        vector<cv::Point3d> rvec_vec=land_mark.getRvec_vec();
        vector<cv::Point3d> tvec_vec=land_mark.getTvec_vec();

        vector<pair<int,vector<cv::Point2d>>> each_frame_features=land_mark.getEach_frame_feature();
        vector<pair<int,vector<cv::Point2d>>> each_frame_inliers_features=land_mark.getEach_frame_inliers_features();
        // vector<pair<int,vector<int>>> unordered_list_of_keypoints=land_mark.getUnordered_list_of_keypoints();
        vector<pair<int,vector<int>>> unordered_list_of_keypoints=land_mark.getUnordered_inliers_list_of_keypoints();

        //size of keyframes 
        int rt_vector_size = rvec_vec.size();
        int points_size=temp_local_BA_id_list.size();

        printf("ORB Local_BA :: mappoint size : %d temp_local_BA_size : %d \n",mappoint.size(),points_size);


        Eigen::MatrixXd rvec_eig_local(3,rt_vector_size);//vector of 'rvec' 
        Eigen::MatrixXd tvec_eig_local(3,rt_vector_size);//vector of 'tvec'

        //rotation, translation vector assignments
        for(int i=0;i<rt_vector_size;i++){
            //assign rvec_vec, tvec_Vec to Eigen variables
            rvec_eig_local(0,i)=rvec_vec[i].x;
            rvec_eig_local(1,i)=rvec_vec[i].y;
            rvec_eig_local(2,i)=rvec_vec[i].z;

            tvec_eig_local(0,i)=tvec_vec[i].x;
            tvec_eig_local(1,i)=tvec_vec[i].y;
            tvec_eig_local(2,i)=tvec_vec[i].z;

            printf("ORB Local_BA :: %lf %lf %lf, %lf %lf %lf\n",rvec_vec[i].x, rvec_vec[i].y, rvec_vec[i].z, tvec_vec[i].x, tvec_vec[i].y, tvec_vec[i].z);
        }

        Eigen::Vector2d BA_2d_points_eig(2); // 각 프레임의, 2D features Points
        Eigen::MatrixXd BA_3d_points_eig(3, points_size); // 현재 프레임의 3d points

        for(int i=0; i<points_size;i++){
            // assign mappoint to Eigen variables
            int id=temp_local_BA_id_list[i];
            BA_3d_points_eig(0,i)=mappoint[id].x;
            BA_3d_points_eig(1,i)=mappoint[id].y;
            BA_3d_points_eig(2,i)=mappoint[id].z;
            // printf("id : %d    %lf %lf %lf\n",id,BA_3d_points_eig(0,i),BA_3d_points_eig(1,i),BA_3d_points_eig(2,i));
        }
        
        ceres::Problem problem2;
        // unorderes_list_of_keypoints :: keypoints id of each keyframe
        for(int i=0;i<points_size;i++){
            //3d land mark assign(num 1 ~ num N) 
            int temp_id=temp_local_BA_id_list[i]; // id of 3d points 

            //assign 2d_features_points of each keyframes
            for(int j=0;j<rt_vector_size;j++){
                int curr_keyframe=(keyframeNum-rt_vector_size+j+1);

                int a=0;
                bool match=false;
                int k_index=0;
                int temp_size=unordered_list_of_keypoints.at(curr_keyframe).second.size();

                for(int k=0;k<temp_size;k++){
                    a=unordered_list_of_keypoints.at(curr_keyframe).second[k];

                    if(temp_id==a){
                        match=true;
                        k_index=k;
                        break;
                    }
                }

                if(match==false){
                    continue;
                }

                else{
                    BA_2d_points_eig[0]=each_frame_inliers_features[curr_keyframe].second.at(k_index).x;
                    BA_2d_points_eig[1]=each_frame_inliers_features[curr_keyframe].second.at(k_index).y;

                    Eigen::Vector3d rvec_eig;
                    Eigen::Vector3d tvec_eig;

                    rvec_eig[0]=rvec_vec[j].x; 
                    rvec_eig[1]=rvec_vec[j].y; 
                    rvec_eig[2]=rvec_vec[j].z;

                    tvec_eig[0]=tvec_vec[j].x; 
                    tvec_eig[1]=tvec_vec[j].y; 
                    tvec_eig[2]=tvec_vec[j].z;

                    ceres::CostFunction* cost_function3 = 
                    new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local_pose_fixed, 2, 3>(
                        new SnavelyReprojectionError_Local_pose_fixed(BA_2d_points_eig[0],BA_2d_points_eig[1],focal,pp.x,pp.y,rvec_eig,tvec_eig)
                    );

                    ceres::LossFunction* loss = new ceres::CauchyLoss(1.0);

                    problem2.AddResidualBlock(cost_function3,
                                            loss,
                                            BA_3d_points_eig.col(i).data());                     
                    
                }            
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = 4;
        options.max_num_iterations=100;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem2, &summary);

        cout<<summary.BriefReport()<<endl;
        cout<<"ORB Local_BA :: local BA done\n"<<endl;

        for(int i=0;i<points_size;i++){
            // land_mark (map structure mappoints update )
            int temp_id=temp_local_BA_id_list[i];

            double points3d_1=BA_3d_points_eig(0,i);
            double points3d_2=BA_3d_points_eig(1,i);
            double points3d_3=BA_3d_points_eig(2,i);

            mappoint[temp_id].x=points3d_1;
            mappoint[temp_id].y=points3d_2;
            mappoint[temp_id].z=points3d_3;

            cv::Point3d points_3d=cv::Point3d(points3d_1,points3d_2,points3d_3);
            land_mark.update_land_mark(temp_id, points_3d);
        }

        //points3d (current 3d points update)
        for(int i=0;i<point3d_world.rows;i++){
            int temp_id=index_of_landmark[i];
                point3d_world.at<double>(i,0)=mappoint[temp_id].x;
                point3d_world.at<double>(i,1)=mappoint[temp_id].y;
                point3d_world.at<double>(i,2)=mappoint[temp_id].z;
        }

    }   








// Full BA  ==================================================
    void Full_BA(Mappoint& land_mark, const double focal, 
                cv::Point2d pp, cv::Mat& rvec, cv::Mat& tvec, 
                cv::Mat& point3d_world, int keyframeNum
                ,vector<int>& index_of_landmark
                ,vector<int>& temp_local_BA_id_list,int count_keypoints){

        map<int,Point3d> mappoint=land_mark.getMappoint(); // 3d map points
        map<int,Point3f>::iterator iter;


        vector<cv::Point3d> rvec_vec=land_mark.getLoopRvec_vec();
        vector<cv::Point3d> tvec_vec=land_mark.getLoopTvec_vec();

        vector<pair<int,vector<cv::Point2d>>> each_frame_features=land_mark.getEach_frame_feature();
        vector<pair<int,vector<cv::Point2d>>> each_frame_inliers_features=land_mark.getEach_frame_inliers_features();
        // vector<pair<int,vector<int>>> unordered_list_of_keypoints=land_mark.getUnordered_list_of_keypoints();
        vector<pair<int,vector<int>>> unordered_list_of_keypoints=land_mark.getUnordered_inliers_list_of_keypoints();

        //size of keyframes 
        int rt_vector_size = rvec_vec.size();
        int points_size=temp_local_BA_id_list.size();

        cout<<" Full BA :: rvec size : "<<rt_vector_size<<"tvec size : "<<tvec_vec.size()<<"curr key frame : "<<keyframeNum<<endl;
        printf(" Full BA :: mappoint size : %d temp_local_BA_size : %d \n",mappoint.size(), points_size);

        Eigen::MatrixXd rvec_eig_local(3,rt_vector_size);//vector of 'rvec' 
        Eigen::MatrixXd tvec_eig_local(3,rt_vector_size);//vector of 'tvec'

        //rotation, translation vector assignments
        for(int i=0;i<rt_vector_size;i++){
            rvec_eig_local(0,i)=rvec_vec[i].x;
            rvec_eig_local(1,i)=rvec_vec[i].y;
            rvec_eig_local(2,i)=rvec_vec[i].z;

            tvec_eig_local(0,i)=tvec_vec[i].x;
            tvec_eig_local(1,i)=tvec_vec[i].y;
            tvec_eig_local(2,i)=tvec_vec[i].z;
            // printf("%lf %lf %lf, %lf %lf %lf\n",rvec_vec[i].x, rvec_vec[i].y, rvec_vec[i].z, tvec_vec[i].x, tvec_vec[i].y, tvec_vec[i].z);
        }

        printf(" Full BA :: Rt vector assignment finish \n");
        Eigen::Vector2d BA_2d_points_eig(2); // 각 프레임의, 2D features Points
        Eigen::MatrixXd BA_3d_points_eig(3, points_size); // 현재 프레임의 3d points

        // assign mappoint to Eigen variables
        for(int i=0; i<points_size;i++){
            int id=temp_local_BA_id_list[i];
            BA_3d_points_eig(0,i)=mappoint[id].x;
            BA_3d_points_eig(1,i)=mappoint[id].y;
            BA_3d_points_eig(2,i)=mappoint[id].z;
            // printf("id : %d    %lf %lf %lf\n",id,BA_3d_points_eig(0,i),BA_3d_points_eig(1,i),BA_3d_points_eig(2,i));
        }
        printf(" Full BA :: mappoint assignment finish \n");

        ceres::Problem problem2;
        // unorderes_list_of_keypoints :: keypoints id of each keyframe
        for(int i=0;i<points_size;i++){
            //3d land mark assign(num 1 ~ num N) 
            int temp_id=temp_local_BA_id_list[i]; // id of 3d points 

            //assign 2d_features_points of each keyframes
            for(int j=0;j<rt_vector_size;j++){
                int curr_keyframe=(keyframeNum-rt_vector_size+j+1);
                int a=0;
                bool match=false;
                int k_index=0;
                int temp_size=unordered_list_of_keypoints.at(curr_keyframe).second.size();

                for(int k=0;k<temp_size;k++){
                    a=unordered_list_of_keypoints.at(curr_keyframe).second[k];

                    if(temp_id==a){
                        match=true;
                        k_index=k;
                        break;
                    }
                }

                if(match==false){
                    continue;
                }

                else{
                    BA_2d_points_eig[0]=each_frame_inliers_features[curr_keyframe].second.at(k_index).x;
                    BA_2d_points_eig[1]=each_frame_inliers_features[curr_keyframe].second.at(k_index).y;

                    Eigen::Vector3d rvec_eig;
                    Eigen::Vector3d tvec_eig;

                    rvec_eig[0]=rvec_vec[j].x; 
                    rvec_eig[1]=rvec_vec[j].y; 
                    rvec_eig[2]=rvec_vec[j].z;

                    tvec_eig[0]=tvec_vec[j].x; 
                    tvec_eig[1]=tvec_vec[j].y; 
                    tvec_eig[2]=tvec_vec[j].z;

                    ceres::CostFunction* cost_function3 = 
                    new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local_pose_fixed, 2, 3>(
                        new SnavelyReprojectionError_Local_pose_fixed(BA_2d_points_eig[0],BA_2d_points_eig[1],focal,pp.x,pp.y,rvec_eig,tvec_eig)
                    );

                    ceres::LossFunction* loss = new ceres::CauchyLoss(1.0);

                    problem2.AddResidualBlock(cost_function3,
                                            loss,
                                            BA_3d_points_eig.col(i).data());                     
                }            
            }
        }
        printf(" Full BA :: ceres assignment finish \n");

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = 4;
        options.max_num_iterations=200;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem2, &summary);

        cout<<summary.BriefReport()<<endl;
        cout<<"Full BA done\n"<<endl;

        for(int i=0;i<points_size;i++){
            // land_mark (map structure mappoints update )
            int temp_id=temp_local_BA_id_list[i];

            double points3d_1=BA_3d_points_eig(0,i);
            double points3d_2=BA_3d_points_eig(1,i);
            double points3d_3=BA_3d_points_eig(2,i);

            mappoint[temp_id].x=points3d_1;
            mappoint[temp_id].y=points3d_2;
            mappoint[temp_id].z=points3d_3;

            cv::Point3d points_3d=cv::Point3d(points3d_1,points3d_2,points3d_3);
            land_mark.update_land_mark(temp_id, points_3d);
        }

        //points3d (current 3d points update)

        for(int i=0;i<point3d_world.rows;i++){
            int temp_id=index_of_landmark[i];
                point3d_world.at<double>(i,0)=mappoint[temp_id].x;
                point3d_world.at<double>(i,1)=mappoint[temp_id].y;
                point3d_world.at<double>(i,2)=mappoint[temp_id].z;
        }

        land_mark.clear_loop_rt();
    }


}