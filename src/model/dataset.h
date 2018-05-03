#ifndef DATASET_H
#define DATASET_H

#include "mnist/mnist_reader.hpp"
#include <map>

typedef std::vector<std::vector<double>> container;

template <class Tensor>
struct grouping {
    std::map<int, Tensor> data;

    grouping(container &images, std::vector<unsigned char> &labels){

        std::map<int, std::vector<Tensor>> _data;
        int n = images.size(); 
        int length;
        Tensor I(0, 0);
        int label;

        n = std::min(n, 500);
        std::cout << "truncate: " << n << "\n";

        for(int i=0; i < n; i++){
            length = images[i].size();
            double *im = &(images[i][0]);
            I = Tensor::from_array(im, length, 1);
            label = labels[i];
            _data[label].push_back(I);
        }

        for(int label=0; label < 10; label++){
            data[label] = hcat(_data[label]);
            std::cout << data[label].rows << "x" << data[label].cols << "\n";
        }
    }
};

template <class Tensor>
Tensor toTensor(container &images){
    std::vector<Tensor> vs;
    Tensor I;
    for(auto image: images){
        double *img = &(image[0]);
        I = Tensor::from_array(img, image.size(), 1);
        vs.push_back(I);
    }
    return hcat(vs);
}

#endif
