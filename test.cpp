#include <ctime>
#include <iostream>

#include "dlib/dnn.h"
#include "dlib/image_io.h"
#include "dlib/matrix.h"
namespace dlib {
template <long num_filters, typename SUBNET>
using con5d = dlib::con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET>
using con5 = dlib::con<num_filters, 5, 5, 1, 1, SUBNET>;
template <typename SUBNET>
using downsampler = dlib::relu<dlib::affine<
    con5d<32, dlib::relu<dlib::affine<
                  con5d<32, dlib::relu<dlib::affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET>
using rcon5 = dlib::relu<dlib::affine<con5<45, SUBNET>>>;
using net_type = dlib::loss_mmod<
    dlib::con<1, 9, 9, 1, 1,
              rcon5<rcon5<rcon5<downsampler<
                  dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;
}  // namespace dlib

int main(int argc, char* argv[])
{
    std::cout << "dlibtest" << std::endl;
    dlib::matrix<double> m1(7, 7);
    dlib::matrix<double> m2(7, 2);
    std::cout << m1 * m2 << std::endl;
    std::cout << "hello" << std::endl;
    if (argc == 1) {
        return 0;
    }
    std::cout << "load image:" << argv[2] << " time:" << std::time(nullptr)
              << std::endl;
    dlib::matrix<dlib::rgb_pixel> img;
    load_image(img, argv[2]);
    std::cout << "load mod:" << argv[1] << " time:" << std::time(nullptr)
              << std::endl;
    dlib::net_type net;
    dlib::deserialize(argv[1]) >> net;
    std::cout << "calculate, time:" << std::time(nullptr) << std::endl;
    auto dets = net(img);
    std::cout << "number of faces detected:" << dets.size()
              << " time:" << std::time(nullptr) << std::endl;
    return 0;
}
