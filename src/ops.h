#ifndef OPS_H
#define OPS_H

#include <cmath>

#include "tensor.h"


/*
Tensor operator+(const Tensor A, const Tensor B){}

Tensor operator-(const Tensor A, const Tensor B){}

Tensor operator*(const Tensor A, const Tensor B){}

Tensor operator*(const double k, const Tensor A){}

Tensor operator*(const Tensor A, const double k){ return k*A; }

tuple<Tensor, Tensor, Tensor> bidiagonalize(const Tensor A){

}

tuple<Tensor, Tensor, Tensor> diagonalize(const Tensor A){

}

tuple<Tensor, Tensor, Tensor> SVD(const Tensor A){

}
*/

#define CPU_ELEMENT_WISE(op)                                      \
    CPUTensor operator op(const CPUTensor A, const CPUTensor B){  \
        /* Assertions */                                          \
        assert (A.rows == B.rows and A.cols == B.cols);           \
        CPUTensor C(A.rows, A.cols);                              \
                                                                  \
        /* Settling for two dimensions now */                     \
        for(int i = 0; i < A.rows; i++){                          \
            for(int j=0; j < A.cols; j++){                        \
                C(i, j) = A(i, j) op B(i, j);                     \
            }                                                     \
        }                                                         \
        return C;                                                 \
    }

CPU_ELEMENT_WISE(+)
CPU_ELEMENT_WISE(-)
CPU_ELEMENT_WISE(*)
CPU_ELEMENT_WISE(/)

namespace ops {
    CPUTensor mul(const CPUTensor A, const CPUTensor B){
        assert ( A.cols == B.rows );
        CPUTensor C(A.rows, B.cols);
        for(int i=0; i < A.rows; i++){
            for(int j=0; j < B.cols; j++){
                for(int k=0; k < A.cols; k++){
                    C(i, j) += A(i, k)*B(k, j);
                }
            }
        }
        return C;
    }

    CPUTensor slice(const CPUTensor A, block b){
        if ( b.row.end == -1) b.row.end = A.rows;
        if ( b.col.end == -1) b.col.end = A.cols;

        if ( b.row.start == -1) b.row.start = 0;
        if ( b.col.start == -1) b.col.start = 0;

        CPUTensor C(b.row.size(), b.col.size());
        for(int i=b.row.start; i<b.row.end; i++){
            for(int j=b.col.start; j<b.col.end; j++){
                C(i-b.row.start, j-b.col.start) = A(i, j);
            }
        }
        return C;

    }

    double dot(const CPUTensor A, const CPUTensor B){
//        std::cout << "in dot function\n";
        assert (A.rows == B.rows);
//        std::cout << "assert passed\n";
        double ans=0;
        for(int i=0;i < A.rows; i++) {
            ans+=A(i,0)*B(i,0);
//            std::cout << A(i, 0)*B(i, 0) << " added to ans, is now, " << ans << std::endl;
        }
//        std::cout << "ans calced as " << ans << std::endl;
        return ans;
    }
    
    double norm2(const CPUTensor A) {
//        std::cout << "in norm2 function\n";
        double d = dot(A, A);
//        std::cout << "dot pdct recvd in norm2 as " << d << std::endl;
        return d;
    }
    
    double norm(CPUTensor A) {
//        std::cout << "in norm function\n";
        CPUTensor flattened = A.flatten();
//        print_m(flattened);
//        std::cout << "flattened\n";
        double n2 = norm2(flattened);
//        std::cout << "norm2 rcvd in norm as " << n2 << std::endl;
        double n = sqrt(n2);
//        std::cout << "norm calced as " << n << std::endl;
        return n;
    }
}

#endif
