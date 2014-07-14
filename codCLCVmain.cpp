#include "oclutil.h"

#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <stdexcept>
#include <iostream>

int main(int argc, char** argv)
{
    bool withOCLUtil;
    argc == 1 ? withOCLUtil = true : withOCLUtil = false;
    withOCLUtil ? std::cout << "Usando classe OCLUtil" << std::endl : std::cout << "Usando OCL padrão" << std::endl;

    std::string filename = "imgProc.cl";
    std::ifstream sourceFile(filename.c_str());
    if(sourceFile.fail())
        std::cout<<"Failed to open OpenCL source file"<<std::endl;

    std::string sourceCode(
                std::istreambuf_iterator<char>(sourceFile),
                (std::istreambuf_iterator<char>()));

    cv::Mat image1 = cv::imread("./alaor.jpg");
    cv::Mat image2 = cv::imread("./silentlives.jpg");

    if (withOCLUtil)
    {
        OCLutil ocl(CL_DEVICE_TYPE_GPU,"imgProc.cl","","paracinza,sub",2);

        ocl.CarregarCVMat(image1, 1, 0, false);
        ocl.CarregarCVMat(image2, 1, 1, false);

        image1.convertTo(image1,CV_32FC3);
        image2.convertTo(image2,CV_32FC3);
        cv::cvtColor(image1,image1,CV_BGR2RGBA);
        cv::cvtColor(image2,image2,CV_BGR2RGBA);

        cv::Mat imgSaida(image1.size(),CV_32FC3);
        cv::cvtColor(imgSaida,imgSaida,CV_BGR2RGBA);

        ocl.CarregarCVMat(imgSaida, 1, 2, true);
        
        float sums[image1.rows];
        memset(sums, 0.f, sizeof(sums));

        ocl.CarregarBuffer(sums,image1.rows, 1, 3, true);

        ocl.Exec(1,cl::NDRange(image1.cols, image1.rows),cl::NullRange);

        ocl.LerBufferImg(imgSaida, 2);
        ocl.LerBuffer(sums,image1.rows, 2);

        for(int i = 0;i<image1.rows;i++){
            std::cout<<"sum["<<i<<"]= "<<sums[i]<<std::endl;
        }

        cv::cvtColor(image1,image1,CV_RGBA2BGR);
        cv::cvtColor(imgSaida,imgSaida,CV_RGBA2BGR);

        imgSaida.convertTo(imgSaida,CV_8UC3);
        image1.convertTo(image1,CV_8UC3);


        cv::imshow("entrada",image1);
        cv::imshow("saida",imgSaida);
    }
    else 
    {
        cl_device_type type = CL_DEVICE_TYPE_GPU;
        VECTOR_CLASS<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if(platforms.size() == 0){
            std::cout<<"No OpenCL platforms were found"<<std::endl;
            return EXIT_FAILURE;
        }

        int platformID = -1;

        for(unsigned int i = 0; i < platforms.size(); i++) {
            try {
                VECTOR_CLASS<cl::Device> devices;
                platforms[i].getDevices(type, &devices);
                platformID = i;
                break;
            } catch(std::exception& e) {
                std::cout<<"Error ao ler plataforma: "<<std::endl;
                continue;
            }
        }

        if(platformID == -1){
            std::cout<<"No compatible OpenCL platform found"<<std::endl;
        }

        cl::Context context;
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
        } catch(std::exception& e) {
            std::cout<<"Failed to create an OpenCL context!"<<std::endl;
            return EXIT_FAILURE;
        }

        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

        // Make program of the source code in the context
        cl::Program program = cl::Program(context, source);

        VECTOR_CLASS<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        std::string buildOptions="";

        // Build program for these specific devices
        cl_int error = program.build(devices, buildOptions.c_str());
        if(error != 0) {
            std::cout << "Build log:" << std::endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
            return EXIT_FAILURE;
        }

        cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

        image1.convertTo(image1,CV_32FC3);
        image2.convertTo(image2,CV_32FC3);
        cv::cvtColor(image1,image1,CV_BGR2RGBA);
        cv::cvtColor(image2,image2,CV_BGR2RGBA);

        cv::Mat imgSaida(image1.size(),CV_32FC3);
        cv::cvtColor(imgSaida,imgSaida,CV_BGR2RGBA);

        cl::Image2D clImage1 = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          cl::ImageFormat(CL_RGBA,CL_FLOAT), image1.cols,
                                          image1.rows, 0, image1.data);
        cl::Image2D clResult = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                                          cl::ImageFormat(CL_RGBA,CL_FLOAT), image1.cols,
                                          image1.rows, 0, imgSaida.data);

        cl::Kernel paracinza = cl::Kernel(program, "paracinza");

        cl::Image2D clImage2 = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    cl::ImageFormat(CL_RGBA,CL_FLOAT), image2.cols,
                                    image2.rows, 0, image2.data);

        float sums[image1.rows];
        memset(sums, 0.f, sizeof(sums));

        cl::Buffer clsums = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float)*(image1.rows), sums);

        cl::Kernel subImg = cl::Kernel(program, "sub");

        subImg.setArg(0,clImage1);
        subImg.setArg(1,clImage2);
        subImg.setArg(2,clResult);
        subImg.setArg(3,clsums);

        error = queue.enqueueNDRangeKernel(
            subImg,
            cl::NullRange,
            cl::NDRange(image1.cols, image1.rows),
            cl::NullRange
        );

        if(error != 0){
            std::cout <<"Error a executar: "<<error<< std::endl;
            return EXIT_FAILURE;
        }

        cl::size_t<3> origin;
        origin[0] = 0;origin[1] = 0;origin[2] = 0;

        cl::size_t<3> region;
        region[0] = image1.cols;region[1] = image1.rows;region[2] = 1;

        error = queue.enqueueReadImage(clResult, CL_TRUE,
                                origin, region, 0, 0,
                                imgSaida.data, NULL, NULL);

        if(error != 0){
            std::cout <<"Error ao ler: "<<error<< std::endl;
            return EXIT_FAILURE;
        }

        queue.enqueueReadBuffer(clsums,CL_TRUE,0,sizeof(float)*(image1.rows),sums);

        for(int i = 0;i<image1.rows;i++){
            std::cout<<"sum["<<i<<"]= "<<sums[i]<<std::endl;
        }

        cv::cvtColor(image1,image1,CV_RGBA2BGR);
        cv::cvtColor(imgSaida,imgSaida,CV_RGBA2BGR);

        imgSaida.convertTo(imgSaida,CV_8UC3);
        image1.convertTo(image1,CV_8UC3);

        cv::imshow("entrada",image1);
        cv::imshow("saida",imgSaida);
    }
    
    cv::waitKey();
    return EXIT_SUCCESS;
}