_NVCC='nvcc'
NVFLAGS='-lcublas'
CXXFLAGS='-std=c++11'
DEBUGFLAGS='-g'

NVCC=$(_NVCC) $(NVFLAGS) $(CXXFLAGS) $(DEBUGFLAGS)

test:
	$(NVCC) test.cpp

test_cuda:
	$(NVCC) test_cuda.cpp
