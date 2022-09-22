#include "kvl_pe.h"



namespace LocalMapping{
    void Erase_OverLab(vector<int>& local_BA_id_list, vector<int>& temp_local_BA_id_list)
    {
        temp_local_BA_id_list=local_BA_id_list;
        sort(temp_local_BA_id_list.begin(),temp_local_BA_id_list.end());
        temp_local_BA_id_list.erase(unique(temp_local_BA_id_list.begin(),temp_local_BA_id_list.end()),temp_local_BA_id_list.end());
    }

    void Erase_FirstKeyframe(vector<int>& id_list, vector<int>& size_list, int size1, int vec_size, int local_ba_frame)
    {
        if(vec_size==local_ba_frame){
            int zahnm=size_list.front();
            vector<int> temp_id_list;
            for(int i=zahnm;i<size1;i++){
                temp_id_list.push_back(id_list[i]);
            }
            size_list.erase(size_list.begin());

            id_list=temp_id_list;
        }
    }

    // insert index_of_landmark into local_BA_id_list
    // insert size1 into local_BA_points_size_list
    void Insert_LocalList(vector<int>& local_BA_id_list, vector<int>& index_of_landmark, 
                                    vector<int>& local_BA_points_size_list, int size1)
    {
        for(int i=0;i<size1;i++){
            local_BA_id_list.push_back(index_of_landmark[i]);
        }
        local_BA_points_size_list.push_back(size1);            
    }


}