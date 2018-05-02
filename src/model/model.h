#ifndef MODEL_H
#define MODEL_H

#include <float.h>
#include <map>
#include <vector>
#include "tensor/indexing.h"
#include "linalg/linalg.h"
#include "model/dataset.h"


template <class Tensor>
struct decomposition {
    Tensor U, Sigma, V;
};

template <class Tensor>
struct model {
    std::map<int, decomposition<Tensor>> params;
    model(dataset<Tensor> &ds){
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

    int classify(Tensor T){
        int k = 10;
        /* 
         * Assume the columns of tensor contain the flattened image.
         * Now use matrix multiplications with U, and back, then norm to
         * compute the loss.
         * Take argmin of loss. Implement argmin at the GPU? Speedup.
         */
        Tensor alpha;
        Tensor Uk;
        Tensor z;

        int digit;
        double min_loss = DBL_MAX;
        double loss;
        for(int i = 0; i < 10; i++) {
            auto decomp = params[i];
            block kb = block(0, decomp.U.rows)(0, k);
            Uk = slice(decomp.U, kb);
            alpha = Uk.transpose()*T;
            z = Uk*alpha;
              
            loss = norm(z-T);
            if(loss < min_loss) {
                min_loss = loss;
                digit = i;
            }
        }
        return digit;
    }
};

#endif
