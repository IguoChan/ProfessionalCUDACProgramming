CU_APPS=simpleDivergence simpleDeviceQuery sumMatrix
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -g -G -arch=sm_89 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}