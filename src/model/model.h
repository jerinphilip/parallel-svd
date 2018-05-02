#ifndef MODEL_H
#define MODEL_H

#include <map>

template <class Tensor>
struct decomposition {
    Tensor U, Sigma, V;
};

template <class Tensor>
struct model {
    std::map<int, decomposition<Tensor>> params;

};

#endif
