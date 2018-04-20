#include "ops.h"
#include "indexing.h"

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
CPU_ELEMENT_WISE(/)

// CPU_ELEMENT_WISE(*)

bool operator==(const CPUTensor A, const CPUTensor B){
    return (A - B).is_zero();
}

CPUTensor operator*(const CPUTensor A, const CPUTensor B){
    assert ( A.cols == B.rows );
    CPUTensor C(A.rows, B.cols);
    for(int i=0; i < A.rows; i++){
        for(int j=0; j < B.cols; j++){
            C(i, j) = 0;
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
    assert (A.rows == B.rows);

    double ans=0;
    for(int i=0;i < A.rows; i++) {
        ans+=A(i,0)*B(i,0);
    }
    return ans;
}

double norm2(const CPUTensor A) {
    double d = dot(A, A);
    return d;
}

double norm(CPUTensor A) {
    CPUTensor flattened = A.flatten();
    double n2 = norm2(flattened);
    double n = sqrt(n2);
    return n;
}

CPUTensor operator*(CPUTensor A, double s) {
    for(int i = 0; i < A.rows; i++) {
        for(int j = 0; j < A.cols; j++) {
            A(i, j) *= s;
        }
    }
    return A;
}

CPUTensor operator*(double s, CPUTensor A) {
    return A*s;
}


