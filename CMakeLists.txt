cmake_minimum_required(VERSION 3.5)

#project information
project(kvl_PE)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/thirdparty/g2o/cmake_modules)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(OpenCV_DIR [OpenCV.cmake PATH])


find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin 0.4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(g2o REQUIRED)





set(HDRS
  include/DBoW2/BowVector.h           include/DBoW2/FBrief.h
  include/DBoW2/QueryResults.h        include/DBoW2/TemplatedDatabase.h   include/DBoW2/FORB.h
  include/DBoW2/DBoW2.h               include/DBoW2/FClass.h              include/DBoW2/FeatureVector.h
  include/DBoW2/ScoringObject.h       include/DBoW2/TemplatedVocabulary.h)
set(SRCS 
  src/BowVector.cpp     src/FBrief.cpp        src/FORB.cpp
  src/FeatureVector.cpp src/QueryResults.cpp  src/ScoringObject.cpp)
set(G2O_LIBS
	g2o_core 
	g2o_types_sba 
	g2o_types_slam3d
	g2o_types_sim3
	g2o_solver_csparse 
	g2o_stuff 
	g2o_csparse_extension
	)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIR})
include_directories(
${Pangolin_INCLUDE_DIRS}   
${G2O_INCLUDE_DIR}
${CSPARSE_INCLUDE_DIR}
  /usr/include/suitesparse
)

#KVL tracker based Pose estimation 

set(SOURCE_FILES kvl_pe.cpp)

add_executable(kvl_PE ${SOURCE_FILES})
target_link_libraries(kvl_PE ${OpenCV_LIBRARIES} ${G2O_LIBS} g2o_types_slam3d ${CSPARSE_LIBRARY})
target_link_libraries(kvl_PE Ceres::ceres)
target_link_libraries(kvl_PE pango_display pango_python )
target_link_libraries(kvl_PE
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/build/libDBoW2.so
)



