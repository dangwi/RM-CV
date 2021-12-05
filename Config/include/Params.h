//
// Created by liming on 12/26/17.
//

#ifndef __PARAMS_H__
#define __PARAMS_H__

#include <cstdio>
#include <cstdlib>

#include <iostream>

using namespace std;

class Params {
public:
    Params() = default;

    int LoadCalibration(const string &calibration);

    int LoadConfig(const string &config);
    
    int LoadBuffConfig(const string &buffconfig);
    
    float armour_dir_angle;
    float camera_width;
    float camera_height;
    float camera_fps;
    float camera_fx;
    float camera_fy;
    float camera_cx;
    float camera_cy;
    float camera_k1;
    float camera_k2;
    float camera_k3;

    int blur_sz;
    float blur_sigma;

    int hmin, hmax, smin, smax, vmin, vmax;
    int blue_hmin, blue_hmax, red_hmin, red_hmax,red_hmin1, red_hmax1;
    int contours_length_min, contours_length_max;
    float LED_ratio_min, LED_ratio_max;
    float LED_width_min, LED_width_max;
    float marker_parallel_angle;
    float marker_vertical_angle;
    float marker_direction_angle;
    float marker_ratio_min, marker_ratio_max;
    float marker_size_min, marker_size_max;

    int transformer_template_width, transformer_template_height;
    float transformer_template_score_thres;
    int transformer_hmin;
    int transformer_hmax;
    int transformer_gray_min;
    int transformer_gray_max;
    int transformer_area_min;
    int transformer_area_max;
    float transformer_c2_s_ratio_min;
    float transformer_c2_s_ratio_max;
    float transformer_ellipse_epsi;
    float transformer_ellipse_inlier_ratio;
    float transformer_ellipse_radius;
    float transformer_big_marker_size;
    float transformer_small_marker_size;

    int target_color;
    int target_size;

    float target_bubing_shift_x;
    float target_bubing_shift_y;
    float target_bubing_shift_z;
    float target_bubing_L;

    float target_shaobing_shift_x;
    float target_shaobing_shift_y;
    float target_shaobing_shift_z;
    float target_shaobing_L;

    float target_yingxiong_shift_x;
    float target_yingxiong_shift_y;
    float target_yingxiong_shift_z;
    float target_yingxiong_L;

    int dilateSize;
    int closeSize;
    float smallDelayTime;
    float bigDelayTime;

    float buffSimilaritymin;
    float buffhwratiomin;
    float buffhwratiomax;
    float buffContourAreamin;
    float buffContourAreamax;

    float centerContourmin;
    float centerSimilarity;
    float centerhwration;

    float yoffset;
    float xoffset;

    int buffhmin, buffhmax, buffsmin, buffsmax, buffvmin, buffvmax;
    int buffblue_hmin, buffblue_hmax, buffred_hmin, buffred_hmax;

    //anti top
    float x_change;
    float t_multi;
};

#endif  //__PARAMS_H__
