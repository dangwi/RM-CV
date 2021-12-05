//
// Created by Finn on 2021/11/9.
//

#ifndef YOLOX_DETECTOR_HPP
#define YOLOX_DETECTOR_HPP
#include"TRTModule.hpp"
#include<opencv2/opencv.hpp>
#include<vector>
#include<mutex>
#include<chrono>
#include <math.h>
#include "setting.h"
#include "Params.h"
#include "communicator.h"
#include "Anti_top.h"

using namespace std;
using namespace cv;

struct GridAndStride
{
    int gridx;
    int gridy;
    int stride;
};

struct Object
{
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    int label;
    float conf;
};

class LED {
public:
	Point2f center;
	float dir_angle;
	float len;
};

class Marker {
public:
	LED LEDs[2];
	Point2f kpts[4];
	Point2f center;
	Marker(Object& obj) {
		LEDs[0].center.x = (obj.x1 + obj.x2) / 2;
		LEDs[0].center.y = (obj.y1 + obj.y2) / 2;
		LEDs[0].len = sqrt(pow(obj.y1 - obj.y2, 2) + pow(obj.x1 - obj.x2, 2));
		LEDs[0].dir_angle = atan((obj.x1 - obj.x2) / (obj.y2 - obj.y1));
		LEDs[1].center.x = (obj.x3 + obj.x4) / 2;
		LEDs[1].center.y = (obj.y3 + obj.y4) / 2;
		LEDs[1].len = sqrt(pow(obj.y3 - obj.y4, 2) + pow(obj.x3 - obj.x4, 2));
		LEDs[1].dir_angle = atan((obj.x4 - obj.x3) / (obj.y3 - obj.y4));
		kpts[0].x = obj.x1;
		kpts[0].x = obj.y1;
		kpts[1].x = obj.x2;
		kpts[1].x = obj.y2;
		kpts[2].x = obj.x3;
		kpts[2].x = obj.y3;
		kpts[3].x = obj.x4;
		kpts[3].x = obj.y4;
		center = (LEDs[0].center + LEDs[1].center) * 0.5f;
	}
};

class Detector{
private:
	Params params;
	Anti_top anti_top;
	COMMUNICATOR* communicator = nullptr;
    Setting *setting;
    static constexpr int IMG_W = 1280;
    static constexpr int IMG_H = 1000;
    static constexpr int INPUT_H=512;
    static constexpr int INPUT_W=640;
    static constexpr int NUM_CLASSES = 36;
    static constexpr float CONF_THRESH = 0.6;
    static constexpr float NMS_THRESH = 0.45;
    const std::vector<int> strides = {8, 16, 32};
    TRTModule trt_model;
    int OUTSIZE;

    mutex input_lock;
    mutex preds_lock;

    class Pkg2Infer{
    public:
        cv::Mat img = cv::Mat(512, 640, CV_8UC3, cv::Scalar(114, 114, 114));
        bool updated = false;
        chrono::time_point<chrono::system_clock> time_stamp;
    } pkg2infer;

    class Pkg2Post{
    public:
        cv::Mat img = cv::Mat(512, 640, CV_8UC3, cv::Scalar(114, 114, 114));
        bool updated = false;
        chrono::time_point<chrono::system_clock> time_stamp;
        float* preds;
    } pkg2post;

    //前处理相关
    int unpad_h;
    int unpad_w;
    cv::Mat src_img;
    cv::Mat resized_img;

    //推理相关
    float* input_data;//存放最终的网络输入数据，构造中申请内存，析构中释放

    //后处理相关
    vector<GridAndStride> grid_strides;
    std::vector<Object> proposals;//置信度筛选后的目标
    std::vector<Object> objects;//nms后的目标
    double scale_ratio;
    float* post_data;//后处理的数据，构造中申请内存，析构中释放


    void generate_grids_and_stride(const std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);

    float* blobFromImage(cv::Mat& img, float* blob);

	bool armour_process(float& x, float& y, float& z);
	float get_armour_len(Marker& marker);
    void qsort_descent_inplace(std::vector<Object>& objects, int left, int right);
    void qsort_descent_inplace(std::vector<Object>& objects);

    bool is_overlap(const Object& a, const Object& b);

    void generate_proposals(std::vector<GridAndStride>& grid_strides, float* preds, float prob_threshold, std::vector<Object>& proposals);
    
    void nms(const std::vector<Object>& proposals, std::vector<int>& picked, float nms_threshold);

    void draw_img(cv::Mat& img,std::vector<Object>& objects);

public:
    Detector(Setting* _setting, COMMUNICATOR* _communicator);
    ~Detector();
    void pre_process();//前处理
    void do_inference();//前处理
    void post_process();//后处理



};

#endif //YOLOX_DETECTOR_HPP
