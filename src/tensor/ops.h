#ifndef OPS_H
#define OPS_H

#include <cmath>
#include "tensor.h"
#include "indexing.h"

#define declare_ops(_tensor)                         \
    _tensor operator+(const _tensor, const _tensor); \
    _tensor operator-(const _tensor, const _tensor); \
    _tensor operator*(const _tensor, const _tensor); \
    _tensor operator*(const double, const _tensor);  \
    _tensor operator*(const _tensor, const double);  \
    _tensor operator/(const _tensor, const _tensor); \
    bool operator==(const _tensor, const _tensor);   \
    _tensor slice(const _tensor, block b);           \
    double norm2(const _tensor);                     \
    double norm(const _tensor);                      \
    double dot(const _tensor, const _tensor);        \
    _tensor hcat(std::vector<_tensor>);              \
    void set_slice(_tensor&, block, _tensor&);

declare_ops(CPUTensor);
declare_ops(CUDATensor);
CUDATensor transpose(CUDATensor);

#endif
