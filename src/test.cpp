#include "storage.h"
#include "tensor.h"
#include "utils.h"
#include "ops.h"

int main(int argc, char *argv[]){
    int m, n;
    m = 5, n = 5;
    CPUTensor A(m, n), B(m, n);
    A = random(m, n);
    B = random(m, n);
    block s = block()(1, 3)(0, -1);
    print_m(ops::slice(A, s));
    print_m(A);
    print_m(B);
    print_m(A + B);
    print_m(A - B);
    print_m(A * B);
    print_m(A / B);
    print_m(ops::mul(A, B));
    return 0;
}
