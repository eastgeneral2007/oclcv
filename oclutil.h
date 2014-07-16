#ifndef OCLUTIL_H
#define OCLUTIL_H

#include <CL/cl.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>

class OCLutil
{
public:
    OCLutil(cl_device_type type, std::string aqr, std::string buildOptions, std::string nomeRot, int n);
    void Exec(int indexRot, cl::NDRange tamGlobal, cl::NDRange tamLoc);
    void CarregarCVMatf(cv::Mat cvMat, int indexRot, int indexParam, bool escrita);
    void CarregarCVMatui(cv::Mat cvMat, int indexRot, int indexParam, bool escrita);
    void CarregarBuffer(float *buffer, int tam, int indexRot, int indexParam, bool escrita);
    void CarregarBuffer(int *buffer, int tam, int indexRot, int indexParam, bool escrita);
    void CarregarFloat(float f,int indexRot,int indexParam);
    void CarregarInt(int in,int indexRot,int indexParam);
    void LerBufferImgf(cv::Mat &cvMat, int indexParam);
    void LerBufferImgui(cv::Mat &cvMat, int indexParam);
    void LerBuffer(float *buffer, int tam, int indexParam);
    void LerBuffer(int *buffer, int tam, int indexParam);
private:
    std::vector<cl::Kernel> rotina;
    cl::CommandQueue queue;
    cl::Context context;
    std::vector<cl::Image2D>climg;
    std::vector<cl::Buffer>clbffer;
    std::vector<cv::Mat>cvImg;
};

#endif // OCLUTIL_H
