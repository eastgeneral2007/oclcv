#include "oclutil.h"
#include <CL/cl.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/ocl/ocl.hpp>


OCLutil::OCLutil(cl_device_type type,std::string arq,std::string buildOptions,std::string nomeRot,int n)
{

    VECTOR_CLASS<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if(platforms.size() == 0){
        std::cout<<"No OpenCL platforms were found"<<std::endl;
    }

    int platformID = -1;

    for(unsigned int i = 0; i < platforms.size(); i++) {
        try {
            VECTOR_CLASS<cl::Device> devices;
            platforms[i].getDevices(type, &devices);
            platformID = i;
            break;
        } catch(std::exception e) {
            std::cout<<"Error ao ler plataforma: "<<std::endl;
            continue;
        }
    }


    if(platformID == -1){
        std::cout<<"No compatible OpenCL platform found"<<std::endl;
    }

    cl::Platform platform = platforms[platformID];
    std::cout << "Using platform vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;


    // Use the preferred platform and create a context
    cl_context_properties cps[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platform)(),
        0
    };

    try {
        context = cl::Context(type, cps);
    } catch(std::exception e) {
        std::cout<<"Failed to create an OpenCL context!"<<std::endl;
    }

    std::string filename = arq;
    std::ifstream sourceFile(filename.c_str());
    if(sourceFile.fail())
        std::cout<<"Failed to open OpenCL source file"<<std::endl;

    std::string sourceCode(
                std::istreambuf_iterator<char>(sourceFile),
                (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

    // Make program of the source code in the context
    cl::Program program = cl::Program(context, source);

    VECTOR_CLASS<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    std::string deviceInfo;
    cl_ulong memInfo;
    size_t tam;
    cl_uint clUnit;
    int indexDev = 0;
    int maxU = 0;
    for (int i = 0; i < devices.size(); ++i)
    {
        devices[i].getInfo((cl_device_info) CL_DEVICE_NAME, &deviceInfo);
        std::cout << "Device info: " << deviceInfo << std::endl;
        devices[i].getInfo((cl_device_info) CL_DEVICE_VERSION, &deviceInfo);
        std::cout << "Versão CL: " << deviceInfo << std::endl;
        devices[i].getInfo((cl_device_info) CL_DRIVER_VERSION, &deviceInfo);
        std::cout << "Versão Driver: " << deviceInfo << std::endl;
        devices[i].getInfo((cl_device_info) CL_DEVICE_GLOBAL_MEM_SIZE, &memInfo);
        std::cout << "Memoria Global: " << memInfo << std::endl;
        devices[i].getInfo((cl_device_info) CL_DEVICE_LOCAL_MEM_SIZE, &memInfo);
        std::cout << "Memoria Local: " << memInfo << std::endl;
        devices[i].getInfo((cl_device_info) CL_DEVICE_LOCAL_MEM_SIZE, &tam);
        std::cout << "Max tamanho Work-group: " << tam << std::endl;
        devices[i].getInfo((cl_device_info) CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &clUnit);
        std::cout << "Max dimensao: " << clUnit << std::endl;
        devices[i].getInfo((cl_device_info) CL_DEVICE_MAX_COMPUTE_UNITS, &clUnit);
        std::cout << "Unidades CL: " << clUnit << std::endl;
        std::cout << "*********************************" << std::endl;

        if((int)clUnit>maxU){
            indexDev = i;
            maxU = (int)clUnit;
        }
    }

    // Build program for these specific devices
    cl_int error = program.build(devices, buildOptions.c_str());
    if(error != 0) {
        std::cout << "Build log:" << std::endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
    }

    std::cout << "Index Dispositino selecionado: " << indexDev << std::endl;
    queue = cl::CommandQueue(context, devices[indexDev]);

    int posi = 0;
    int posf = 0;
    for(int i = 0; i < n; i++){

        posf = nomeRot.find(",",posi);
        std::string nomeRoti;
        if(posf != -1){
            nomeRoti = nomeRot.substr(posi,posf-posi);
        }else{
            nomeRoti = nomeRot.substr(posi);
        }
        std::cout<<"Nome rotina["<<i<<"]: "<<nomeRoti.data()<<std::endl;
        rotina.push_back(cl::Kernel(program, nomeRoti.data()));
        posi = posf + 1;
    }
}

void OCLutil::CarregarCVMatf(cv::Mat cvMat, int indexRot, int indexParam, bool escrita){

    cl_mem_flags flags = 0;
    if(escrita){
        flags = CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR;
    }else{
        flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    }

    cvMat.convertTo(cvMat,CV_32FC3);
    cv::cvtColor(cvMat,cvMat,CV_BGR2RGBA);

    cvImg.push_back(cvMat);

    cl::Image2D clImage = cl::Image2D(context, flags,
                                      cl::ImageFormat(CL_RGBA,CL_FLOAT), cvImg.back().cols,
                                      cvImg.back().rows, 0, cvImg.back().data);

    climg.push_back(clImage);

    rotina[indexRot].setArg(indexParam,clImage);
}

void OCLutil::CarregarCVMatui(cv::Mat cvMat, int indexRot, int indexParam, bool escrita){

    cl_mem_flags flags = 0;
    if(escrita){
        flags = CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR;
    }else{
        flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    }

    cv::cvtColor(cvMat,cvMat,CV_BGR2RGBA);

    cvImg.push_back(cvMat);

    cl::Image2D clImage = cl::Image2D(context, flags,
                                      cl::ImageFormat(CL_RGBA,CL_UNORM_INT8), cvImg.back().cols,
                                      cvImg.back().rows, 0, cvImg.back().data);

    climg.push_back(clImage);

    rotina[indexRot].setArg(indexParam,clImage);
}

void OCLutil::Exec(int indexRot,cl::NDRange tamGlobal,cl::NDRange tamLoc){
    cl_int error = queue.enqueueNDRangeKernel(
                rotina[indexRot],
                cl::NullRange,
                tamGlobal,
                tamLoc);

    if(error != 0){
        std::cout <<"Error a executar: "<<error<< std::endl;
    }

}

void OCLutil::CarregarFloat(float f,int indexRot,int indexParam){
    rotina[indexRot].setArg(indexParam,f);
}

void OCLutil::CarregarInt(int in,int indexRot,int indexParam){
    rotina[indexRot].setArg(indexParam,in);
}

void OCLutil::CarregarBuffer(float *buffer,int tam, int indexRot, int indexParam,bool escrita){

    cl_mem_flags flags = 0;
    if(escrita){
        flags = CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR;
    }else{
        flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    }

    cl::Buffer clfloat = cl::Buffer(context, flags,sizeof(float)*tam, buffer);
    rotina[indexRot].setArg(indexParam,clfloat);
    clbffer.push_back(clfloat);
}

void OCLutil::CarregarBuffer(int *buffer,int tam, int indexRot, int indexParam,bool escrita){

    cl_mem_flags flags = 0;
    if(escrita){
        flags = CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR;
    }else{
        flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    }

    cl::Buffer clint = cl::Buffer(context, flags,sizeof(int)*tam, buffer);
    rotina[indexRot].setArg(indexParam,clint);
    clbffer.push_back(clint);
}

void OCLutil::LerBufferImgf(cv::Mat &cvMat, int indexParam){

    cl::size_t<3> origin;
    origin[0] = 0;origin[1] = 0;origin[2] = 0;

    cl::size_t<3> region;
    region[0] = cvMat.cols;region[1] = cvMat.rows;region[2] = 1;

    cl_int error = queue.enqueueReadImage(climg[indexParam], CL_TRUE,
                                          origin, region, 0, 0,
                                          cvImg[indexParam].data, NULL, NULL);

    cvMat = cv::Mat(cvImg[indexParam]);
    cv::cvtColor(cvMat,cvMat,CV_RGBA2BGR);
    cvMat.convertTo(cvMat,CV_8UC3);

    if(error != 0){
        std::cout <<"Error ao ler: "<<error<< std::endl;
    }
}

void OCLutil::LerBufferImgui(cv::Mat &cvMat, int indexParam){

    cl::size_t<3> origin;
    origin[0] = 0;origin[1] = 0;origin[2] = 0;

    cl::size_t<3> region;
    region[0] = cvMat.cols;region[1] = cvMat.rows;region[2] = 1;

    cl_int error = queue.enqueueReadImage(climg[indexParam], CL_TRUE,
                                          origin, region, 0, 0,
                                          cvImg[indexParam].data, NULL, NULL);

    cvMat = cv::Mat(cvImg[indexParam]);
    cv::cvtColor(cvMat,cvMat,CV_RGBA2BGR);

    if(error != 0){
        std::cout <<"Error ao ler: "<<error<< std::endl;
    }
}

void OCLutil::LerBuffer(float *buffer, int tam, int indexParam){
    queue.enqueueReadBuffer(clbffer[indexParam],CL_TRUE,0,sizeof(float)*(tam),buffer);
}

void OCLutil::LerBuffer(int *buffer, int tam, int indexParam){
    queue.enqueueReadBuffer(clbffer[indexParam],CL_TRUE,0,sizeof(float)*(tam),buffer);
}
