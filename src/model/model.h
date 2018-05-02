#ifndef MODEL_H
#define MODEL_H

#include <map>
#include <vector>

template <class Tensor>
struct decomposition {
    Tensor U, Sigma, V;
};

template <class Tensor>
struct model {
    std::map<int, decomposition<Tensor>> params;

    std::vector<int> classify(Tensor T){
        /* 
         * Assume the columns of tensor contain the flattened image.
         * Now use matrix multiplications with U, and back, then norm to
         * compute the loss.
         * Take argmin of loss. Implement argmin at the GPU? Speedup.
         */

    }

};

#endif
