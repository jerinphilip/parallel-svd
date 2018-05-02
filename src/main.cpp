#include <cassert>
#include "storage/storage.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "tensor/indexing.h"
#include "utils/utils.h"
#include "linalg/linalg.h"
#include "model/model.h"
#include "model/dataset.h"

int main(int argc, char *argv[]){
	const char *MNIST_DATA_LOCATION = argv[1];
	std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

	// Load MNIST data
	auto ds = mnist::read_dataset<std::vector, std::vector, double, uint8_t>(MNIST_DATA_LOCATION);
    auto training = dataset<CUDATensor>(ds.training_images, ds.training_labels);
    auto classifier = model<CUDATensor>(training);
    return 0;
}
