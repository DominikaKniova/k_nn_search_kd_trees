cmake_minimum_required(VERSION 3.12)
project(project)

set(CMAKE_C++_STANDARD 11)

include_directories(.)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})


add_executable(project
        kd_tree.cpp
        data_generator.cpp
        data_generator.h
        kd_node.h
        kd_tree.h
        point.h
        main.cpp types.h naive_search.h naive_search.cpp utils.h utils.cpp point.cpp comparators.h comparators.cpp data.h data.cpp visualizer.h visualizer.cpp leaf_points.h leaf_points.cpp search_structures.h search_structures.cpp search_comparator.h search_comparator.cpp binomial_heap.h)

if(VTK_LIBRARIES)
    target_link_libraries(project ${VTK_LIBRARIES})
else()
    target_link_libraries(project vtkHybrid vtkWidgets)
endif()
