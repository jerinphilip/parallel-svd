#ifndef OPS_H
#define OPS_H

#include <cmath>
#include "tensor.h"
#include "indexing.h"

CPUTensor operator+(const CPUTensor, const CPUTensor);
CPUTensor operator-(const CPUTensor, const CPUTensor);
CPUTensor operator*(const CPUTensor, const CPUTensor);
CPUTensor operator*(const double, const CPUTensor);
CPUTensor operator*(const CPUTensor, const double);
CPUTensor operator/(const CPUTensor, const CPUTensor);
bool operator==(const CPUTensor, const CPUTensor);

CPUTensor slice(const CPUTensor, block b);
double norm2(const CPUTensor);
double norm(const CPUTensor);
double dot(const CPUTensor, const CPUTensor);


#endif
