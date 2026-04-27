# No hace falta find_package aquí porque ya se hizo en el principal
add_library(utils image.cc dct.cc)

# Enlazamos las librerías necesarias para la parte de utilidades
target_link_libraries(utils 
    PNG::PNG 
    JPEG::JPEG 
    OpenMP::OpenMP_CXX
)
