CU_APPS=simpleDivergence simpleDeviceQuery sumMatrix reduceInteger \
	nestedHelloWorld nestedHelloWorldNew nestedReduce nestedReduceNosync \
	nestedReduce2
C_APPS=

all: ${C_APPS} ${CU_APPS}

nestedHelloWorld: nestedHelloWorld.cu
	nvcc -g -G -arch=sm_89 -rdc=true -o $@ $<
nestedHelloWorldNew: nestedHelloWorldNew.cu
	nvcc -g -G -arch=sm_89 -rdc=true -o $@ $<
nestedReduce: nestedReduce.cu
	nvcc -g -G -arch=sm_89 -rdc=true -DCUDA_FORCE_CDP1_IF_SUPPORTED -o $@ $<
nestedReduceNosync: nestedReduceNosync.cu
	nvcc -g -G -arch=sm_89 -rdc=true -DCUDA_FORCE_CDP1_IF_SUPPORTED -o $@ $<
nestedReduce2: nestedReduce2.cu
	nvcc -g -G -arch=sm_89 -rdc=true -DCUDA_FORCE_CDP1_IF_SUPPORTED -o $@ $<
%: %.cu
	nvcc -g -G -arch=sm_89 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
