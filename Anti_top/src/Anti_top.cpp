#include <cmath>
#include <iostream>
#include "Anti_top.h"
#define PI 3.1415926f
using namespace std;

float t_delay = 0.05; // 拨弹盘会有0.05s的延迟
static float center_x_temp = 0; //用于中心坐标的一阶低通滤波
float center_k = 0.01; //用于中心坐标的一阶低通滤波
float turn_head_k = 0.6; // 提前转头参数拟合


//清除反陀螺相关数据
void Anti_top::anti_top_clear(){
	center_x_temp = 0;
    center_x = 0.f; 
    last_rotate_angle = 0.f; 
    t_angle.clear(); 
    anti_top_delta_t = 0.f;
    init_flag = 0;
    shoot_flag = 0;
    turn_head_flag = 0;
    shoot_t1 = 0.f;
    shoot_t2 = 0.f;
    angle_speed = 0.f;
}

void Anti_top::anti_top_run_A(float& X, float& Z, float fly_t){
    cout<<"Plan_A"<<endl;
    // 如果某两帧转角突变大小大于45度，而且这两帧转角的角度为一正一负，则认为发生了突变
    if (fabs(last_rotate_angle-cur_rotate_angle)>PI/4.f && ((last_rotate_angle<0&&cur_rotate_angle>0)||(last_rotate_angle>0&&cur_rotate_angle<0))){
        cout<<"突变转角大小："<<fabs(last_rotate_angle - cur_rotate_angle)*180.f/PI<<endl;
        if (init_flag){
            gettimeofday(&anti_top_t2, nullptr);
            anti_top_delta_t = static_cast<float>(static_cast<double>(anti_top_t2.tv_sec - anti_top_t1.tv_sec) + static_cast<double>(anti_top_t2.tv_usec - anti_top_t1.tv_usec) * 1e-6);
            gettimeofday(&anti_top_t1, nullptr);

            if (t_angle.size()>1) {
                cv::fitLine(t_angle, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);
                cv::Point2f point0;
                point0.x = line_para[2]; //横坐标为时间坐标
                point0.y = line_para[3]; //纵坐标为转角坐标
                float k = line_para[1] / line_para[0]; //线的斜率
                float shoot_angle = shoot_armour_dist/R+car_v*fly_t;
                if (k>0){
                    shoot_t1 = (-shoot_angle-point0.y)/k+point0.x;
                    shoot_t2 = (shoot_angle-point0.y)/k+point0.x;
                }
                else if (k<0){
                    shoot_t1 = (shoot_angle-point0.y)/k+point0.x;
                    shoot_t2 = (-shoot_angle-point0.y)/k+point0.x;
                }
                //cout<<"曲线拟合结果:"<<point0.x<<' '<<point0.y<<' '<<k<<' '<<shoot_t1<<' '<<shoot_t2<<endl;
            }

            initial_angle = PI/2 - cur_rotate_angle;
            angle_speed = (cur_rotate_angle-last_rotate_angle)/anti_top_delta_t;
            cout<<"R = "<<R<<"   "<<"w = "<<angle_speed<<"   "<<"周期 = "<<anti_top_delta_t<<endl;
            t_angle.clear();
        }
    }
    last_rotate_angle = cur_rotate_angle;

    // 如果满足切换到Plan_B的条件
    if(anti_top_delta_t>t_multi*fly_t) { 
        cout<<"切换到反陀螺Plan_B"<<endl;
        return;
    }

    if (init_flag){
        gettimeofday(&cur_t, nullptr);
        auto cur_t_s = static_cast<float>(static_cast<double>(cur_t.tv_sec - anti_top_t1.tv_sec) + static_cast<double>(cur_t.tv_usec - anti_top_t1.tv_usec) * 1e-6);
        cout<<"历时="<<cur_t_s<<"   "<<"R = "<<R<<"   "<<"w = "<<angle_speed<<"   "<<"周期 = "<<anti_top_delta_t<<"子弹飞行时间:"<<fly_t<<endl;
        if (cur_t_s > 3*anti_top_delta_t){
            anti_top_clear();
            return;
        }
        X-=R*sin(cur_rotate_angle); // 枪管指向车的中心
        // 车的中心的一阶低通滤波
        center_x = (1 - center_k) * center_x_temp + center_k * X;
        center_x_temp = center_x;

        X-=car_v*Z*fly_t; // 运动预测
        t_angle.push_back(cv::Point2f(cur_t_s,  cur_rotate_angle)); // 记录转角和时间的数据
        if (shoot_t1 && shoot_t2){
            while ((cur_t_s+(fly_t+t_delay))>anti_top_delta_t){ // 击打n个周期后的装甲板,（n>=1）。
                cur_t_s-=anti_top_delta_t;
            }

            if ((cur_t_s+(fly_t+t_delay))>shoot_t1 && (cur_t_s+(fly_t+t_delay))<shoot_t2){ // 如果子弹飞行时间位于指定阈值内，则发射子弹
                Z=-Z;
                cout<<"shoot now!"<<endl;
            }
        }
    }
}

void Anti_top::anti_top_run_B(float& X, float& Z, float fly_t){
    cout<<"Plan_B"<<endl;
    if (last_rotate_angle == 0.f) last_rotate_angle = cur_rotate_angle; //last_rotate_angle初始化

    // 如果某两帧转角突变大小大于45度，而且这两帧转角的角度为一正一负，则认为发生了突变
    if (fabs(last_rotate_angle-cur_rotate_angle)>PI/4.f && ((last_rotate_angle<0&&cur_rotate_angle>0)||(last_rotate_angle>0&&cur_rotate_angle<0))){
        if (init_flag){  //如果之前已经突变了一次，则可以计算出转动角速度
            turn_head_flag = 0; //提前转头标志位置为0
            shoot_flag = 1; // 射击标志位置为1
            gettimeofday(&anti_top_t2, nullptr); //记录时间节点anti_top_t2
            anti_top_delta_t = static_cast<float>(static_cast<double>(anti_top_t2.tv_sec - anti_top_t1.tv_sec) + static_cast<double>(anti_top_t2.tv_usec - anti_top_t1.tv_usec) * 1e-6); //计算两次突变时间间隔
            gettimeofday(&anti_top_t1, nullptr); //记录时间节点anti_top_t1
            /*if(anti_top_delta_t>t_multi*fly_t) {  //不满足切换到Plan_A的条件，继续维持Plan_B
                t_angle.clear();
            }*/

            // 计算当前装甲板在上一次突变时的位置
            if (cur_rotate_angle < 0){
                cur_rotate_angle1 -=PI/2; 
            }
            else {
                cur_rotate_angle1 += PI/2;
            }

            // 用于后续提前转头
            initial_angle = PI/2 - cur_rotate_angle;
            // 计算角速度
            angle_speed = (cur_rotate_angle1-cur_rotate_angle)/anti_top_delta_t;
            // 记录当前转角，用于后续计算角速度
            cur_rotate_angle1 = cur_rotate_angle;
            cout<<"R = "<<R<<"   "<<"w = "<<angle_speed<<"   "<<"周期 = "<<anti_top_delta_t<<endl;
        }
        else{  // 如果第一次突变，则还无法计算出转动角速度
            gettimeofday(&anti_top_t1, nullptr); // 记录时间节点anti_top_t1
            cur_rotate_angle1 = cur_rotate_angle; // 记录当前转角，用于后续计算角速度
            init_flag = 1; // 并置初始化位为1
        }
    }
    last_rotate_angle = cur_rotate_angle;
    center_x = X-R*sin(cur_rotate_angle); // 计算车的中心x坐标
    // 由于计算的对面车的中心坐标波动较大，所以使用低通滤波
    center_x = (1 - center_k) * center_x_temp + center_k * center_x;
    center_x_temp = center_x;

    // 如果满足切换到Plan_A的条件
    /*if (anti_top_delta_t!=0 && anti_top_delta_t<t_multi*fly_t) { 
        if (t_angle.size()>1) { //t_angle至少有两个点才能进行直线拟合，否则报错
            cv::fitLine(t_angle, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);
            cv::Point2f point0;
            point0.x = line_para[2]; //横坐标为时间坐标
            point0.y = line_para[3]; //纵坐标为转角坐标
            float k = line_para[1] / line_para[0]; //线的斜率
            float shoot_angle = shoot_armour_dist/R+car_v*fly_t;
            if (k>0){
                shoot_t1 = (-shoot_angle-point0.y)/k+point0.x;
                shoot_t2 = (shoot_angle-point0.y)/k+point0.x;
            }
            else if (k<0){
                shoot_t1 = (shoot_angle-point0.y)/k+point0.x;
                shoot_t2 = (-shoot_angle-point0.y)/k+point0.x;
            }
        }
        t_angle.clear();
        cout<<"切换到反陀螺Plan_A"<<endl;
        return;
    }*/

    if (init_flag){
        gettimeofday(&cur_t, nullptr); // 获取当前时刻
        auto cur_t_s = static_cast<float>(static_cast<double>(cur_t.tv_sec - anti_top_t1.tv_sec) + static_cast<double>(cur_t.tv_usec - anti_top_t1.tv_usec) * 1e-6);
        //t_angle.push_back(cv::Point2f(cur_t_s,  cur_rotate_angle)); // 记录Plan_A的数据，随时准备切换到Plan_A
        if (anti_top_delta_t){
            cout<<"历时="<<cur_t_s<<"   "<<"R = "<<R<<"   "<<"w = "<<angle_speed<<"   "<<"周期 = "<<anti_top_delta_t<<"子弹飞行时间:"<<fly_t<<endl;
            float delta_theta = angle_speed * (fly_t + t_delay); //子弹飞行过程中装甲板转过的角度
            while (fabs(delta_theta)>=PI/2.0){
                if (delta_theta<0) delta_theta+=PI/2.0;
                else delta_theta-=PI/2.0;
            }
            float cur_theta = PI/2 - cur_rotate_angle; 
            if ((angle_speed<0 && (cur_theta+delta_theta)<PI/4.0) || (angle_speed>0 && (cur_theta+delta_theta)>(3.0*PI/4.0))){
                if (delta_theta>0){
                    delta_theta = -(PI/2 - delta_theta);
                }
                else if (delta_theta<0){
                    delta_theta = PI/2 + delta_theta;
                }
            }
            X -= (Z*car_v*fly_t+R*(cos(cur_theta) - cos(cur_theta+delta_theta))); // 第一项为运动预测补偿，第二项为转动补偿
            if (shoot_flag){
                float shoot_x = plan_b_x1;
                if (R==hero_R) shoot_x = plan_b_x2;
                if (X<(shoot_x) && X>(-shoot_x)){ // 如果枪管位置处于设定阈值内，则发射
                    Z=-Z; // z置为负数代表发射子弹
                    cout<<"shoot now!!!"<<endl;
                }
            }
        }
    }
}

void Anti_top::anti_top_run(float& X, float& Z, float fly_t){
    car_v = 0; //这一句如果注释掉就是开启运动预测的反陀螺
    /*if (anti_top_delta_t && anti_top_delta_t<t_multi*fly_t){ //如果对面转动非常快或者距离较远（子弹飞行时间fly_t较大），则切换到Plan_A。
        anti_top_run_A(X,Z,fly_t);
    }*/
    anti_top_run_B(X,Z,fly_t); //否则使用Plan_B
}
