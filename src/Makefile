_NVCC='nvcc'
NVFLAGS='-lcublas'
CXXFLAGS="-std=c++11" "-I ."
DEBUGFLAGS='-g'

NVCC=$(_NVCC) $(NVFLAGS) $(CXXFLAGS) $(DEBUGFLAGS)

all: test main

test: build/libtensor.so build/libstorage.so build/libutils.so 
	$(NVCC) -Xcompiler -L./build/ -ltensor -lstorage  -lutils test.cpp -o build/test

build/libstorage.so: storage/*.cpp
	$(NVCC) -Xcompiler -fPIC  storage/*.cpp -shared -o build/libstorage.so

build/libtensor.so:  build/libstorage.so tensor/*.cpp
	$(NVCC) -Xcompiler -fPIC -L./build/  -shared -o build/libtensor.so tensor/*.cpp

#build/liblinalg.so: build/libtensor.so linalg/*.cpp
#	$(NVCC) -Xcompiler -fPIC -L./build  -shared -o build/liblinalg.so linalg/*.cpp
#


build/libutils.so: build/libtensor.so utils/*.cpp
	$(NVCC) -Xcompiler -fPIC -L./build -shared -o build/libutils.so utils/*.cpp

clean:
	rm build/*
	rm a.out

main: build/libtensor.so build/libstorage.so build/libutils.so
	$(NVCC) -Xcompiler -L./build/ -ltensor -lstorage  -lutils main.cpp -o build/main

.PHONY: clean
