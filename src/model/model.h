#ifndef MODEL_H
#define MODEL_H

#include <map>
#include <vector>
#include "linalg/linalg.h"
#include "model/dataset.h"

template <class Tensor>
struct decomposition {
    Tensor U, Sigma, V;
};

template <class Tensor>
struct model {
    std::map<int, decomposition<Tensor>> params;
    model(dataset &ds){
        for(auto p: ds.data){
            std::cout << "SVD : " << p.first << "\n";
            auto dt = svd(p.second);
            decomposition<Tensor> decomp;
            decomp.U = std::get<0>(dt);
            decomp.Sigma = std::get<1>(dt);
            decomp.V = std::get<2>(dt);
            params[p.first] = decomp;
        }
    }

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
