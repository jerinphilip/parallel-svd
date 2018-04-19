#include "storage.h"
#include "tensor.h"
#include "utils.h"
#include "ops.h"

int main(int argc, char *argv[]){
    CPUTensor A(2, 3), B(2, 3);
    A = random(2, 3);
    B = random(2, 3);
    print_m(A);
    print_m(B);
    print_m(A + B);
    return 0;
}
