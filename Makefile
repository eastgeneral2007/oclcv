all:
	g++ codCLCVmain.cpp oclutil.cpp `pkg-config --libs opencv` -lamdocl64 -o oclcv
clean:
	rm oclcv
