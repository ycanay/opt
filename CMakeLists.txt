cmake_minimum_required(VERSION 3.10)

project(easy_optimization)


set(IPOPT_LIBRARY_DIRS /usr/local/lib)
set(IPOPT_INCLUDE_DIRS /usr/local/include/coin-or)

message("ipopt binary path is: " ${IPOPT_LIBRARY_DIRS} )
message("ipopt include path is: " ${IPOPT_INCLUDE_DIRS} )

include_directories(${IPOPT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
link_directories(${IPOPT_LIBRARY_DIRS})

add_library(optimizer STATIC
    optimization.cpp
)


target_include_directories(optimizer
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(optim 
    main.cpp
)


target_link_libraries(optim
    PRIVATE 
        optimizer
        ipopt
)