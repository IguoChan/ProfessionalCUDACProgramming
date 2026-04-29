TARGET=hello

all: ${TARGET}

%: %.cu
	nvcc -O2 -arch=sm_89 -o $@ $<
clean:
	rm -f ${TARGET}