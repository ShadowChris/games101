#include<cmath>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>
#include<iostream>
using namespace std;
using namespace Eigen;

int main(){

    // Basic Example of cpp
    std::cout << "Example of cpp \n";
    float a = 1.0, b = 2.0;
    std::cout << a << std::endl;
    std::cout << a/b << std::endl;
    std::cout << std::sqrt(b) << std::endl;
    std::cout << std::acos(-1) << std::endl;
    std::cout << std::sin(30.0/180.0*acos(-1)) << std::endl;

    // Example of vector
    std::cout << "Example of vector \n";
    // vector definition
    Eigen::Vector3f v(1.0f,2.0f,3.0f);
    Eigen::Vector3f w(1.0f,0.0f,0.0f);
    // vector output
    std::cout << "Example of output \n";
    std::cout << v << std::endl;
    // vector add
    std::cout << "Example of add \n";
    std::cout << v + w << std::endl;
    // vector scalar multiply
    std::cout << "Example of scalar multiply \n";
    std::cout << v * 3.0f << std::endl;
    std::cout << 2.0f * v << std::endl;

    // Example of matrix
    std::cout << "Example of matrix \n";
    // matrix definition
    Eigen::Matrix3f i,j;
    i << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
    j << 2.0, 3.0, 1.0, 4.0, 6.0, 5.0, 9.0, 7.0, 8.0;
    // matrix output
    std::cout << "Example of output \n";
    std::cout << i << std::endl;

    /**
     * Exercise1
     */
    // matrix add i + j
    std::cout << "matrix add i + j: " << std::endl;
    std::cout << i + j << std::endl;
    // matrix scalar multiply i * 2.0
    std::cout << "matrix scalar multiply i * 2.0: " << std::endl;
    std::cout << i * 2.0 << std::endl;
    // matrix multiply i * j
    std::cout << "matrix multiply i * j: " << std::endl;
    std::cout << i * j << std::endl;
    // matrix multiply vector i * v
    std::cout << "matrix multiply vector i * v: " << std::endl;
    std::cout << i * v << std::endl;

    cout << endl;
    /**
     * Exercise2
     * 描述：给定一个点P=(2,1), 将该点绕原点先逆时针旋转45◦，再平移(1,2), 
     * 计算出变换后点的坐标（要求用齐次坐标进行计算）。
     */
    cout << "Exercise2: " << endl;
    Vector3d p, o, pv;
    // p点
    p << 2,1,1;

    // 变换矩阵
    Matrix3d trMatrix;

    // 弧度制角度
    double r = 45.0 * M_PI / 180;

    trMatrix << cos(r), -sin(r), 1, 
                sin(r), cos(r), 2, 
                0, 0, 1;

    Vector3d res = trMatrix * p;

    cout << "matrix: " << endl << trMatrix << endl;
    cout << "result vector:" << endl << res << endl;
    
    return 0;
}