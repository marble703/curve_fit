cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(curve_fit_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

add_executable(curve_fit src/main.cpp)
target_include_directories(curve_fit PRIVATE ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(curve_fit PRIVATE ${OpenCV_LIBS} ${CERES_LIBRARIES})
