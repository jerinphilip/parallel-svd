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
    Tensor loss(Tensor Z, int k) {
        /* assuming columns of Z are input samples */
        std::vector<double> norms;
        double *norms_array;

        block kb = block(0, (*this).U.rows)(0, k);
        Tensor Uk = slice((*this).U, kb);

        Tensor Z_prime = Uk*Uk.transpose()*Z;
        for (int i = 0; i < Z.cols; i++) {
            block col = block(0, Z.rows)(i, i+1);
            Tensor z = slice(Z, col);
            Tensor z_prime = slice(Z_prime, col);
            double norm_z = norm(z-z_prime);
            norms.push_back(norm_z);
        }
        norms_array = &norms[0];
        Tensor norms_tensor(1, Z.cols);
        norms_tensor.from_array(norms_array, 1, Z.cols);

        /* returns a 1*n (where n is the number of samples) tensor of norms */
        return norms_tensor;
    }
};

template <class Tensor>
struct model {
    std::map<int, decomposition<Tensor>> params;
    model(grouping<Tensor> &ds){
        for(auto p: ds.data){
            std::cout << "SVD : " << p.first << "\n";
            //            print_m(p.second);
            auto dt = svd(p.second);
            decomposition<Tensor> decomp;
            decomp.U = std::get<0>(dt);
            decomp.Sigma = std::get<1>(dt);
            decomp.V = std::get<2>(dt);
            params[p.first] = decomp;
        }
    }

    int classify(Tensor T){
        /* 
         * Assume the columns of tensor contain the flattened image.
         * Now use matrix multiplications with U, and back, then norm to
         * compute the loss.
         * Take argmin of loss. Implement argmin at the GPU? Speedup.
         */
        std::vector<Tensor> norms;
        int k = 10;
        for(int i=0; i<9; i++){
            auto norm_t = params[i].loss(T, k);
            norms.push_back(norm_t);
        }
        return 0;
    }
};

#endif
