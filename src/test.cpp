#include "storage.h"
#include "tensor.h"
#include "utils.h"
#include "ops.h"

int main(int argc, char *argv[]){
    CPUTensor A(2, 2), B(2, 2);
    A = random(2, 2);
    B = random(2, 2);
    print_m(A);
    print_m(B);
    print_m(A + B);
    print_m(A - B);
    print_m(A * B);
    print_m(A / B);
    print_m(ops::mul(A, B));
    return 0;
}
