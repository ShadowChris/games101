#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
/**
 * @brief 
 * 作业参考链接：https://scarletsky.github.io/2020/06/09/games101-notes-transformation/
 * 提高题+代码理解：https://blog.csdn.net/ycrsw/article/details/123834579
 */

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.
    // 实现图形绕z轴旋转的函数，即定义绕z轴的旋转矩阵。
    float radian = rotation_angle / 180 * MY_PI;

    model << cos(radian), -sin(radian), 0, 0,
             sin(radian), cos(radian), 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;

    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.
    // 透视投影：Mp = Mo * Mp->o，Mo：正交投影矩阵；Mp->o：将透视图压缩成正交图
    Eigen::Matrix4f Mpo;

    Mpo << zNear, 0, 0, 0,
           0, zNear, 0, 0,
           0, 0, zNear + zFar, -zNear * zFar,
           0, 0, 1, 0;
    
    Eigen::Matrix4f Mo = Eigen::Matrix4f::Identity();

    // 构建Mo
    float n = zNear, f = zFar;
    float t = n * tan(eye_fov / 2 * MY_PI / 180.0), b = -t;
    float r = aspect_ratio * t, l = -r;

    Mo(0, 0) = 2 / (r - l);
    Mo(1, 1) = 2 / (t - b);
    Mo(2, 2) = 2 / (n - f);

    Mo(0, 3) = -(r + l) / (r - l);
    Mo(1, 3) = -(t + b) / (t - b);
    Mo(2, 3) = -(n + f) / (n - f);

    projection = Mo * Mpo;

    return projection;
}

Eigen::Matrix4f get_rotation(Vector3f axis, float angle) {

}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
