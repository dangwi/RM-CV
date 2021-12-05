//
// Created by helloworld on 2021/3/7.
//

#include <opencv2/opencv.hpp>
#include "Params.h"

int Params::LoadCalibration(const string &calibration) {
    // read calibration parameters
    cv::FileStorage fCalibration(calibration, cv::FileStorage::READ);
    camera_width = fCalibration["Camera.width"];
    camera_height = fCalibration["Camera.height"];
    camera_fps = fCalibration["Camera.fps"];
    camera_fx = fCalibration["Camera.fx"];
    camera_fy = fCalibration["Camera.fy"];
    camera_cx = fCalibration["Camera.cx"];
    camera_cy = fCalibration["Camera.cy"];
    camera_k1 = fCalibration["Camera.k1"];
    camera_k2 = fCalibration["Camera.k2"];
    camera_k3 = fCalibration["Camera.k3"];
    cout << "===Camera Calibration===" << endl;
    cout << "width: " << camera_width << endl;
    cout << "height: " << camera_height << endl;
    cout << "fps: " << camera_fps << endl;
    cout << "fx: " << camera_fx << endl;
    cout << "fy: " << camera_fy << endl;
    cout << "cx: " << camera_cx << endl;
    cout << "cy: " << camera_cy << endl;
    cout << "k1: " << camera_k1 << endl;
    cout << "k2: " << camera_k2 << endl;
    cout << "k3: " << camera_k3 << endl;
    return 0;
}
int Params::LoadConfig(const string &config) {
    // read config parameters
    cv::FileStorage fConfig(config, cv::FileStorage::READ);
    // ImageFrame
    blur_sz = fConfig["ImageFrame.blur_size"];
    blur_sigma = fConfig["ImageFrame.blur_sigma"];
    // Detector
    armour_dir_angle = fConfig["Detector.armour_dir_angle"];
    hmin = fConfig["Detector.hmin"];
    hmax = fConfig["Detector.hmax"];
    blue_hmin = fConfig["Detector.blue_hmin"];
    blue_hmax = fConfig["Detector.blue_hmax"];
    red_hmin = fConfig["Detector.red_hmin"];
    red_hmax = fConfig["Detector.red_hmax"];
    smin = fConfig["Detector.smin"];
    smax = fConfig["Detector.smax"];
    vmin = fConfig["Detector.vmin"];
    vmax = fConfig["Detector.vmax"];
    contours_length_min = fConfig["Detector.contours_length_min"];
    contours_length_max = fConfig["Detector.contours_length_max"];
    LED_ratio_min = fConfig["Detector.LED_ratio_min"];
    LED_ratio_max = fConfig["Detector.LED_ratio_max"];
    LED_width_min = fConfig["Detector.LED_width_min"];
    LED_width_max = fConfig["Detector.LED_width_max"];
    marker_parallel_angle = fConfig["Detector.marker_parallel_angle"];
    marker_vertical_angle = fConfig["Detector.marker_vertical_angle"];
    marker_direction_angle = fConfig["Detector.marker_direction_angle"];
    marker_ratio_min = fConfig["Detector.marker_ratio_min"];
    marker_ratio_max = fConfig["Detector.marker_ratio_max"];
    marker_size_min = fConfig["Detector.marker_size_min"];
    marker_size_max = fConfig["Detector.marker_size_max"];
    // Transformer
    transformer_gray_min = fConfig["Transformer.gray_min"];
    transformer_gray_max = fConfig["Transformer.gray_max"];
    transformer_area_min = fConfig["Transformer.area_min"];
    transformer_area_max = fConfig["Transformer.area_max"];
    transformer_c2_s_ratio_min = fConfig["Transformer.c2_s_ratio_min"];
    transformer_c2_s_ratio_max = fConfig["Transformer.c2_s_ratio_max"];
    transformer_small_marker_size = fConfig["Transformer.small_marker_size"];
    transformer_big_marker_size = fConfig["Transformer.big_marker_size"];
    transformer_ellipse_epsi = fConfig["Transformer.ellipse_epsi"];
    transformer_ellipse_inlier_ratio = fConfig["Transformer.ellipse_inlier_ratio"];
    transformer_ellipse_radius = fConfig["Transformer.ellipse_radius"];
    // target
    target_color = fConfig["Target.color"];
    target_size = fConfig["Target.size"];
    target_bubing_shift_x = fConfig["Target.bubing.shift_x"];
    target_bubing_shift_y = fConfig["Target.bubing.shift_y"];
    target_bubing_shift_z = fConfig["Target.bubing.shift_z"];
    target_bubing_L = fConfig["Target.bubing.L"];
    target_shaobing_shift_x = fConfig["Target.shaobing.shift_x"];
    target_shaobing_shift_y = fConfig["Target.shaobing.shift_y"];
    target_shaobing_shift_z = fConfig["Target.shaobing.shift_z"];
    target_shaobing_L = fConfig["Target.shaobing.L"];
    target_yingxiong_shift_x = fConfig["Target.yingxiong.shift_x"];
    target_yingxiong_shift_y = fConfig["Target.yingxiong.shift_y"];
    target_yingxiong_shift_z = fConfig["Target.yingxiong.shift_z"];
    target_yingxiong_L = fConfig["Target.yingxiong.L"];
    //anti_top
    x_change = fConfig["Anti_top.x_change"];
    t_multi = fConfig["Anti_top.change_t"];
    return 0;
}

int Params::LoadBuffConfig(const string &buffconfig) {
    // read config parameters
    cv::FileStorage fConfig(buffconfig, cv::FileStorage::READ);
    //buff config
    dilateSize = fConfig["dilateSize"];
    closeSize = fConfig["closeSize"];
    smallDelayTime = fConfig["smallDelayTime"];
    bigDelayTime = fConfig["bigDelayTime"];

    buffSimilaritymin = fConfig["buffSimilaritymin"];
    buffhwratiomin = fConfig["buffhwratiomin"];
    buffhwratiomax = fConfig["buffhwratiomax"];
    buffContourAreamin = fConfig["buffContourAreamin"];
    buffContourAreamax = fConfig["buffContourAreamax"];

    centerContourmin = fConfig["centerContourmin"];
    centerSimilarity = fConfig["centerSimilarity"];
    centerhwration = fConfig["centerhwration"];
    yoffset = fConfig["yoffset"];
    xoffset = fConfig["xoffset"];
    buffhmin = fConfig["buffhmin"];
    buffhmax = fConfig["buffhmax"];
    buffsmin = fConfig["buffsmin"];
    buffsmax = fConfig["buffsmax"];
    buffvmin = fConfig["buffvmin"];
    buffvmax = fConfig["buffvmax"];
    buffblue_hmin = fConfig["buffblue_hmin"];
    buffblue_hmax = fConfig["buffblue_hmax"];
    buffred_hmin = fConfig["buffred_hmin"];
    buffred_hmax = fConfig["buffred_hmax"];
    cout << "===Buff parameters===" << endl;
    cout << "dilateSize = " << dilateSize << endl;
    cout << "cloaseSize = " << closeSize << endl;
    cout << "smallDelayTime = " << smallDelayTime << endl;
    cout << "bigDelayTime = " << bigDelayTime << endl;
    cout << "buffSimilarity = " << buffSimilaritymin << endl;
    cout << "buffhwrationmin = " << buffhwratiomin << endl;
    cout << "buffhwrationmax = " << buffhwratiomax << endl;
    cout << "buffContourAreamin = " << buffContourAreamin << endl;
    cout << "buffContourAreamax = " << buffContourAreamax << endl;
    cout << "centerContourmin = " << centerContourmin << endl;
    cout << "centerSimilarity = " << centerSimilarity << endl;
    cout << "centerhwration = " << centerhwration << endl;
    cout << "yoffset = " << yoffset << endl;
    cout << "xoffset = " << xoffset << endl;
    cout << "buffhmin = " << buffhmin << endl;
    cout << "buffhmax = " << buffhmax << endl;
    cout << "buffsmin = " << buffsmin << endl;
    cout << "buffsmax = " << buffsmax << endl;
    cout << "buffvmin = " << buffvmin << endl;
    cout << "buffvmax = " << buffvmax << endl;
    cout << "buffblue_hmin = " << buffblue_hmin << endl;
    cout << "buffblue_hmax = " << buffblue_hmax << endl;
    cout << "buffred_hmin = " << buffred_hmin << endl;
    cout << "buffred_hmax = " << buffred_hmax << endl;

}


