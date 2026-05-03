#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "png.h"
#include <vector>
#include <assert.h>
#include <iostream>
#include <memory>
#include "utils/image.h"
#include "utils/dct.h"
#include <string>
#include <chrono>
#include <future> // Tarea 1.3: Necesario para std::async
#include <omp.h>

/* --- FUNCIONES DE SOPORTE PARA SRM --- */

Image<float> get_srm_3x3() {
    Image<float> kernel(3, 3, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -4); kernel.set(1, 2, 0, 2);
    kernel.set(2, 0, 0, -1); kernel.set(2, 1, 0, 2); kernel.set(2, 2, 0, -1);
    return kernel;
}

Image<float> get_srm_5x5() {
    Image<float> kernel(5, 5, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -2); kernel.set(0, 3, 0, 2); kernel.set(0, 4, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -6); kernel.set(1, 2, 0, 8); kernel.set(1, 3, 0, -6); kernel.set(1, 4, 0, 2);
    kernel.set(2, 0, 0, -2); kernel.set(2, 1, 0, 8); kernel.set(2, 2, 0, -12); kernel.set(2, 3, 0, 8); kernel.set(2, 4, 0, -2);
    kernel.set(3, 0, 0, 2); kernel.set(3, 1, 0, -6); kernel.set(3, 2, 0, 8); kernel.set(3, 3, 0, -6); kernel.set(3, 4, 0, 2);
    kernel.set(4, 0, 0, -1); kernel.set(4, 1, 0, 2); kernel.set(4, 2, 0, -2); kernel.set(4, 3, 0, 2); kernel.set(4, 4, 0, -1);
    return kernel;
}

Image<float> get_srm_kernel(int size) {
    assert(size == 3 || size == 5);
    return (size == 5) ? get_srm_5x5() : get_srm_3x3();
}

/* --- TAREAS PARALELIZABLES --- */

Image<unsigned char> compute_srm(const Image<unsigned char> &image, int kernel_size) {
    auto begin = std::chrono::steady_clock::now();
    std::cout << "Computing SRM " << kernel_size << "x" << kernel_size << "..." << std::endl;          
    Image<float> srm = image.to_grayscale().convert<float>();
    srm = srm.convolution(get_srm_kernel(kernel_size));
    srm = srm.abs().normalized();
    srm = srm * 255;
    Image<unsigned char> result = srm.convert<unsigned char>();
    auto end = std::chrono::steady_clock::now();
    std::cout << "SRM " << kernel_size << "x" << kernel_size << " elapsed time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    return result;
}

Image<unsigned char> compute_ela(const Image<unsigned char> &image, int quality){
    auto begin = std::chrono::steady_clock::now();
    std::cout << "Computing ELA..." << std::endl;
    Image<unsigned char> grayscale = image.to_grayscale();
    save_to_file("_temp.jpg", grayscale, quality);
    Image<float> compressed = load_from_file("_temp.jpg").convert<float>();
    compressed = compressed + (grayscale.convert<float>() * (-1));
    compressed = compressed.abs().normalized() * 255;
    Image<unsigned char> result = compressed.convert<unsigned char>();
    auto end = std::chrono::steady_clock::now();
    std::cout << "ELA elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    return result;
}

Image<unsigned char> compute_dct(const Image<unsigned char> &image, int block_size, bool invert) {
    auto begin = std::chrono::steady_clock::now();
    std::cout << "Computing " << (invert ? "inverse" : "direct") << " DCT " << block_size << "x" << block_size << "..." << std::endl;
    
    Image<float> grayscale = image.convert<float>().to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);

    for(int i = 0; i < (int)blocks.size(); i++){
        float **dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, blocks[i], 0);
        if (invert) {
            for(int k = 0; k < blocks[i].size / 2; k++)
                for(int l = 0; l < blocks[i].size / 2; l++)
                    dctBlock[k][l] = 0.0;
            dct::inverse(blocks[i], dctBlock, 0, 0.0, 255.);
        } else {
            dct::assign(dctBlock, blocks[i], 0);
        }
        dct::delete_matrix(dctBlock);
    }
    
    Image<unsigned char> result = grayscale.convert<unsigned char>();
    auto end = std::chrono::steady_clock::now();
    std::cout << "DCT " << (invert ? "inverse" : "direct") << " elapsed time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    return result;
}

/* --- FUNCIÓN PRINCIPAL --- */

int main(int argc, char **argv) {
    if(argc == 1) {
        std::cerr << "Image filename missing. Usage: ./forense_par <filename>" << std::endl;
        exit(1);
    }

    int block_size = 8;
    Image<unsigned char> image = load_from_file(argv[1]);
	
    // Inicio de la medición global para cálculo de Speed-up
    auto total_begin = std::chrono::steady_clock::now();

    // Tarea 1.3: Paralelismo a nivel de tarea con std::async
    // Se lanzan las 5 tareas de forma asíncrona. Se utiliza std::ref(image) para evitar
    // copias masivas de la imagen original en cada hilo, optimizando la memoria.
    auto f1 = std::async(std::launch::async, compute_srm, std::ref(image), 3);
    auto f2 = std::async(std::launch::async, compute_srm, std::ref(image), 5);
    auto f3 = std::async(std::launch::async, compute_ela, std::ref(image), 90);
    auto f4 = std::async(std::launch::async, compute_dct, std::ref(image), block_size, true);
    auto f5 = std::async(std::launch::async, compute_dct, std::ref(image), block_size, false);

    // .get() recupera el resultado y sincroniza los hilos (barrera de finalización)
    save_to_file("srm_kernel_3x3.png", f1.get());
    save_to_file("srm_kernel_5x5.png", f2.get());
    save_to_file("ela.png", f3.get());
    save_to_file("dct_invert.png", f4.get());
    save_to_file("dct_direct.png", f5.get());

    auto total_end = std::chrono::steady_clock::now();
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "TOTAL PARALLEL EXECUTION TIME: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_begin).count() 
              << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    return 0;
}
