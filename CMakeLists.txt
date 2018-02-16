cmake_minimum_required (VERSION 3.4)

project (heatfast)

set(HDF5_PREFER_PARALLEL ON)
set(CMAKE_C_FLAGS "-Wall -std=c99")

find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_search_module(CONFIG REQUIRED libconfig)

find_library(MATH_LIB m)
find_library(CONFIG_LIB config)

find_path(HDF5_INCLUDE_DIR hdf5.h)
message(STATUS "HDF5 include dir: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 supports parallel: ${HDF5_IS_PARALLEL}")

include_directories(${MPI_INCLUDE_PATH})
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${CONFIG_INCLUDE_DIRS})
include_directories(include)
file(GLOB SOURCES "src/*.c")

add_executable(heatfast ${SOURCES})

target_link_libraries(heatfast "${MPI_LIBRARIES}")
target_link_libraries(heatfast "${MATH_LIB}")
target_link_libraries(heatfast "${HDF5_LIBRARIES}")
target_link_libraries(heatfast "${CONFIG_LIBRARIES}")


target_compile_options(heatfast PUBLIC ${CONFIG_CFLAGS_OTHER})

if(MPI_COMPILE_FLAGS)
	set_target_properties(heatfast PROPERTIES
		COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
	set_target_properties(heatfast PROPERTIES
		LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
