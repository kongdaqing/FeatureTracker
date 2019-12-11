cmake_minimum_required(VERSION 2.8.3)
project(FeatureTracker)


find_package(OpenCV REQUIRED)
add_executable(featureTracker src/AnkerInterface.cpp)
target_link_libraries(featureTracker ${OpenCV_LIBS})


