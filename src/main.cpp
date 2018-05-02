#include <cassert>
#include "storage/storage.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "tensor/indexing.h"
#include "utils/utils.h"
#include "linalg/linalg.h"
#include "mnist/mnist_reader.hpp"
#include <map>

typedef std::vector<std::vector<double>> container;

struct dataset {
    std::map<int, CPUTensor> data;

    dataset(container &images, std::vector<unsigned char> &labels){

        std::map<int, std::vector<CPUTensor>> _data;
        int n = images.size(), length;
        CPUTensor I(0, 0);
        int label;

        for(int i=0; i < n; i++){
            length = images[i].size();
            double *im = &(images[i][0]);
            I = CPUTensor::from_array(im, length, 1);
            label = labels[i];
            _data[label].push_back(I);
        }

        for(int label=0; label < 10; label++){
            data[label] = hcat(_data[label]);
            std::cout << data[label].rows << data[label].cols << "\n";

        }
    }


    
};

int main(int argc, char *argv[]){
	const char *MNIST_DATA_LOCATION = argv[1];
	std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

	// Load MNIST data
	auto ds = mnist::read_dataset<std::vector, std::vector, double, uint8_t>(MNIST_DATA_LOCATION);
    auto training = dataset(ds.training_images, ds.training_labels);
    return 0;
}
