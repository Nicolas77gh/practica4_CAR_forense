cmake_minimum_required(VERSION 3.15)
project(detect)

# 1. BUSCAR PAQUETES 
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# 2. Configuración de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 3. Incluir directorios
include_directories(${PROJECT_SOURCE_DIR})
# Esto le dice dónde están los archivos de cabecera de las librerías
include_directories(${PNG_INCLUDE_DIRS} ${JPEG_INCLUDE_DIRS})

add_subdirectory(utils)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_executable(${PROJECT_NAME} main.cc)

# 4. ENLAZAR
target_link_libraries(${PROJECT_NAME} 
    PNG::PNG 
    JPEG::JPEG 
    utils 
    Threads::Threads 
    OpenMP::OpenMP_CXX
)
