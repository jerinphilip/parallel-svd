#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>


struct Tensor {
    std::vector<int> dims;
    Tensor(std::initializer_list <int> dims): dims(dims) {}

    int _size(){
        int product = 1;
        for(auto d: dims){
            product *= d;
        }
        return product;
    }

};

struct CPUTensor: public Tensor {
    CPUStorage *S;
    CPUTensor(std::initializer_list <int> dims):
        Tensor(dims) {
            S = new CPUStorage(_size());
    }



    friend std::ostream& operator <<(std::ostream &out, const CPUTensor B){
        for(int i=0; i < 2; i++){
            for(int j=0; j < 3; j++){
                out << B.S->data[i*3 + j] << " ";
            }
            out << "\n";
        }
        return out;
    }

    ~CPUTensor(){
        delete S;
    }


};

#endif
