NVCC='nvcc'
NVFLAGS='-lcublas'
CXXFLAGS='-std=c++11'

test:
	$(NVCC) $(NVFLAGS) $(CXXFLAGS)  test.cpp

test_cuda:
	$(NVCC) $(NVFLAGS) $(CXXFLAGS)  test_cuda.cpp
