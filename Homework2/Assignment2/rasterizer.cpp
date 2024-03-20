// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
using namespace std;
/**
 * 作业笔记
 * 1. 作业参考：https://blog.csdn.net/Xuuuuuuuuuuu/article/details/124172397
 * 2. 像素坐标系：原点在左上角，x轴向右，y轴向下；并且为整数
 * 3. 绘制像素：set_pixel()中的x、y需要+0.5，表示像素中心点
 * 4. 标准坐标转重心坐标：computeBarycentric2D()
 * 5. 超采样需要把insideTriangle()中的int改成float，才能在一个像素中判断多个点
 * 6. 超采样如果分4个子点，在三角形内的每个子点的比率为1/4，最后要统计在三角形内的子点个数，再除以4，乘到颜色上
 * 7. 黑边处理：待解决
*/

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

// 超采样需要把原本的int改成float
// static bool insideTriangle(int x, int y, const Vector3f* _v)
static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]

    // 用二维叉积判断两个向量的方向
    Vector3f v01 = _v[1] - _v[0];
    Vector3f v12 = _v[2] - _v[1];
    Vector3f v20 = _v[0] - _v[2];

    Vector3f p(x, y, 1);

    Vector3f v0p = p - _v[0];
    Vector3f v1p = p - _v[1];
    Vector3f v2p = p - _v[2];

    float val0 = v01.cross(v0p).z();
    float val1 = v12.cross(v1p).z();
    float val2 = v20.cross(v2p).z();

    return (val0 > 0 && val1 > 0 && val2 > 0) 
        || (val0 < 0 && val1 < 0 && val2 < 0);
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
// 作业2提高题：https://blog.csdn.net/Xuuuuuuuuuuu/article/details/124172397
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    // v：三角形三个顶点的坐标
    auto v = t.toVector4();
    // cout << "-----start-------" << endl;
    // for (int i = 0; i < v.size(); i++) {
    //     cout <<"v" << i << ": " << v[i] << endl;
    // }
    // cout << endl << "-----end---------" << endl;
    
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle

    // If so, use the following code to get the interpolated z value.
    //auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    //float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    //float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    //z_interpolated *= w_reciprocal;

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.


    // 1 找到三角形的bounding box
    float minXValue = DBL_MAX, minYValue = DBL_MAX, 
    maxXValue = DBL_MIN, maxYValue = DBL_MIN;

    for (int i = 0; i < v.size(); i++) {
        minXValue = min(minXValue, v[i].x());
        minYValue = min(minYValue, v[i].y());
        maxXValue = max(maxXValue, v[i].x());
        maxYValue = max(maxYValue, v[i].y());
    }
    // 2 对bounding box的x，y坐标取整数
    int minX = floor(minXValue);
    int minY = floor(minYValue);
    int maxX = ceil(maxXValue);
    int maxY = ceil(maxYValue);

    vector<vector<float>> superSampling = {
        {0.25, 0.25},
        {0.75, 0.25},
        {0.25, 0.75},
        {0.75, 0.75}
    };
    // 3 遍历bounding box的像素
    for (int x = minX; x <= maxX; x++) {
        for (int y = minY; y <= maxY; y++) {

            // 提高作业：超采样
            int count = 0;
            for (int i = 0; i < superSampling.size(); i++) {
                float x_ss = x + superSampling[i][0];
                float y_ss = y + superSampling[i][1];
                if (insideTriangle(x_ss, y_ss, t.v)) {
                    count++;
                }
            }
            // 如果超采样的像素都不在三角形内，则不绘制
            if (count <= 0) continue;

            // 判断像素是否在三角形内
            if (insideTriangle(x + 0.5, y + 0.5, t.v)) {
                // z-buffer算法
                // computeBarycentric2D：标准坐标转重心坐标。加0.5是为了取像素中心点
                auto[alpha, beta, gamma] = computeBarycentric2D(x + 0.5, y + 0.5, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                // 深度测试
                int index = get_index(x, y);
                if (z_interpolated < depth_buf[index]) {
                    depth_buf[index] = z_interpolated;
                    // Eigen::Vector3f color = t.getColor();
                    // 提高作业：超采样
                    Eigen::Vector3f color = t.getColor() * count / superSampling.size();
                    set_pixel(Eigen::Vector3f(x, y, z_interpolated), color);
                }

            }
        }
    }

}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on