#ifndef ROBOMASTERMARKERDETECTOR3_ANTI_TOP_H
#define ROBOMASTERMARKERDETECTOR3_ANTI_TOP_H

#include <vector>
#include <sys/time.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
using namespace std;

class Anti_top{
    public:
    float shoot_armour_dist1 = 0.04; //Plan_A小装甲板的射击阈值
    float shoot_armour_dist2 = 0.08; //Plan_A大装甲板的射击阈值
    float shoot_armour_dist = shoot_armour_dist1; //根据装甲板类型进行切换，默认初始化为小装甲板阈值
    float infantry_R = 0.22f; // Plan_B对面步兵半径
    float hero_R = 0.27f; // Plan_B对面英雄半径
    float R = infantry_R; // 根据装甲板类型进行切换，默认初始化为步兵半径
    float car_v = 0; // 用于运动预测反陀螺，对面车的平动角速度
    float center_x=0.f; // 对面车的中心坐标
    float angle_speed = 0.f; // 对面车小陀螺的转速(rad / s)
    float cur_rotate_angle = 0.f; // 对面车的当前小陀螺转角
    float cur_rotate_angle1 = 0.f; // 计算需要
    float last_rotate_angle = 0.f; // 对面车上一帧的小陀螺转角
    float t_multi;
    int init_flag = 0; // 初始化标记位
    int shoot_flag = 0; // 自动发射标记位
    timeval cur_t{}; //当前时间
    timeval anti_top_t1{}; //上一次装甲板突变时间
    timeval anti_top_t2{}; //这一次装甲板突变时间
    Anti_top(){}
    void anti_top_run(float& X, float &Z, float fly_t);
    void anti_top_run_A(float& X, float& Z, float fly_t);
    void anti_top_run_B(float& X, float& Z, float fly_t);
    void anti_top_clear();
    private:
    float plan_b_x1 = 0.01; // Plan_B小装甲板的射击阈值
    float plan_b_x2 = 0.02; // Plan_B大装甲板的射击阈值
    float shoot_t1=0.f,shoot_t2=0.f; //Plan_A射击时间区间
    cv::Vec4f line_para;  // Plan_A线性拟合的结果
    vector<cv::Point2f> t_angle; //用于Plan_A
    int turn_head_flag = 0; // 提前转头标记位
    float anti_top_delta_t = 0.f; //装甲板突变周期
    float initial_angle = 0.f;
};
#endif  //ROBOMASTERMARKERDETECTOR3_ANTI_TOP_H
