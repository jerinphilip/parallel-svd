#include <cassert>
#include "storage/storage.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "tensor/indexing.h"
#include "utils/utils.h"
#include "linalg/linalg.h"
#include "mnist/mnist_reader.hpp"

int main(int argc, char *argv[]){
	const char *MNIST_DATA_LOCATION = argv[1];
	std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

	// Load MNIST data
	mnist::MNIST_dataset<std::vector, std::vector<double>, uint8_t> dataset =
		mnist::read_dataset<std::vector, std::vector, double, uint8_t>(MNIST_DATA_LOCATION);
    for (auto p: dataset.training_labels){
        std::cout << (int)p << "\n";
    }
}
