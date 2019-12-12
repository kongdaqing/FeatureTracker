cmake_minimum_required(VERSION 2.8.3)
project(FeatureTracker)


find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(featureTracker src/AnkerInterface.cpp
    src/Anker/readImgDataFile.cpp
    src/ImageFeatureTrack/ImageFeatureTracker.cpp)
target_link_libraries(featureTracker ${OpenCV_LIBS})


