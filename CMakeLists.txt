cmake_minimum_required(VERSION 2.8)
project( EDSR )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( EDSR EDSR.cpp )
target_link_libraries( EDSR ${OpenCV_LIBS} )

