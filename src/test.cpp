#include "storage.h"
#include "tensor.h"

int main(int argc, char *argv[]){
    CPUTensor A({2, 3});
    std::cout << A ;
    return 0;
}
