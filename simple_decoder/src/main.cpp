#include <iostream>

#include <opencv2/core.hpp>

int main()
{
    cv::Mat mymat(2, 2, CV_8U);
    std::cout << mymat << "\n";
    return 0;
}