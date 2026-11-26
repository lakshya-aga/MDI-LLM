cmake_minimum_required(VERSION 3.18)
project(mdi_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Threads REQUIRED)

add_library(mdinode_core STATIC
  src/json.cpp
  src/tensor.cpp
  src/model.cpp
  src/net.cpp
  src/node.cpp)
target_include_directories(mdinode_core PUBLIC src)
target_link_libraries(mdinode_core PUBLIC Threads::Threads)
set_target_properties(mdinode_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(mdinode_core PRIVATE -Wall -Wextra -Wno-unused-parameter)

if(ANDROID)
  # JNI library loaded by the Android app (android/)
  add_library(mdinode SHARED src/jni.cpp)
  target_link_libraries(mdinode PRIVATE mdinode_core log)
else()
  # Command-line node, drop-in replacement for src/secondary.py
  add_executable(mdi_node src/main.cpp)
  target_link_libraries(mdi_node PRIVATE mdinode_core)
endif()
