//
// Created by Finn on 2021/11/9.
//
#include "Detector.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <fstream>
#include <thread>
#include "MVCamera.h"
#include <algorithm>
using namespace std;

Detector::Detector(Setting* _setting, COMMUNICATOR* _communicator):trt_model(INPUT_H,INPUT_W), communicator(_communicator), setting(_setting){
	params.LoadConfig(string("../Config/marker_config.yml"));
	params.LoadCalibration(string("../Config/calibration.yml"));
    scale_ratio = std::min(INPUT_W / (IMG_W*1.0), INPUT_H / (IMG_H*1.0));
    generate_grids_and_stride(strides, grid_strides);
    input_data = new float[3 * INPUT_H * INPUT_W];
    OUTSIZE = trt_model.get_outsize();
    pkg2post.preds = new float[OUTSIZE];
    post_data = new float[OUTSIZE];

    unpad_h = scale_ratio * IMG_H;
    unpad_w = scale_ratio * IMG_W;
    resized_img.create(unpad_h, unpad_w, CV_8UC3);

    //start camera
    MVCamera::Init();
    MVCamera::Play();
    MVCamera::SetExposureTime(false, 7000);
    MVCamera::SetLargeResolution(true);

    //增益
    CameraSetAnalogGain(MVCamera::hCamera, 10);
    CameraSetGamma(MVCamera::hCamera, 50);
}

Detector::~Detector(){
    delete input_data;
    delete pkg2post.preds;
    delete post_data;
}

void Detector::pre_process(){
    while(1)
    {
        //获得图片
        //auto start = chrono::system_clock::now();
        MVCamera::GetFrame(src_img);
        //auto end = chrono::system_clock::now();
        //std::cout << "read_img:"<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms;" << std::endl;
        if(src_img.empty())
            {cout << "no img" << endl;continue;}
        //缩放图片
        auto img_time_stamp = chrono::system_clock::now();
        cv::resize(src_img, resized_img, resized_img.size());
        //更新输入图片
        input_lock.lock();
        resized_img.copyTo(pkg2infer.img(cv::Rect(0, 0, resized_img.cols, resized_img.rows)));
        pkg2infer.updated = true;
        pkg2infer.time_stamp = img_time_stamp;
        input_lock.unlock();
    }

}

void Detector::do_inference(){

    while(1)
    {
        input_lock.lock();
        if(pkg2infer.updated)//如果有新数据就进行推理
        {
            pkg2infer.updated = false;
            //获得输入序列(因为无法控制input_data的使用，因此不允许其他线程修改input_data)
            for (int c = 0; c < 3; c++) {
                for (int h= 0; h < INPUT_H; h++) {
                    for (int w = 0; w < INPUT_W; w++) {
                        input_data[c * INPUT_W * INPUT_H + h * INPUT_W + w] = (float) pkg2infer.img.at<cv::Vec3b>(h, w)[c];
                    }
                }
            }
            #ifdef __SHOWING_IMG__
            Mat temp_img = pkg2infer.img.clone();
            #endif
            auto temp_time_stamp = pkg2infer.time_stamp;

            input_lock.unlock();

            //进行推理
            //auto start = chrono::system_clock::now();
            trt_model(input_data);
            //auto end = chrono::system_clock::now();
            //std::cout << "infer:"<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms;" << std::endl;


            preds_lock.lock();
            #ifdef __SHOWING_IMG__
            pkg2post.img = temp_img.clone();
            #endif
            memcpy(pkg2post.preds, trt_model.preds, OUTSIZE*sizeof(float));
            pkg2post.updated = true;
            pkg2post.time_stamp = temp_time_stamp;
            preds_lock.unlock();


        }
        else//理论上采集图片快于推理，不会进入else
        {
            input_lock.unlock();
            this_thread::yield();//让出cpu时间片
        }
    }
}

auto this_t = chrono::system_clock::now();
auto last_t = chrono::system_clock::now();
void Detector::post_process(){
    while(1)
    {
        preds_lock.lock();
        if(pkg2post.updated)
        {
            //如后处理一定比推理快，则无需此部分cpy 直接对preds指向数据进行操作，操作后解锁即可
            pkg2post.updated=false;
            memcpy(post_data, pkg2post.preds, OUTSIZE*sizeof(float));
            #ifdef __SHOWING_IMG__
            Mat show_img = pkg2post.img.clone();
            #endif
            auto img_time_stamp = pkg2post.time_stamp;
            preds_lock.unlock();

            //后处理
            proposals.clear();
            generate_proposals(grid_strides, post_data,  CONF_THRESH, proposals);
            std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

            qsort_descent_inplace(proposals);
            std::vector<int> picked;
            nms(proposals, picked, NMS_THRESH);

            int count = picked.size();
            objects.resize(count);
            for (int i = 0; i < count; i++)
            {
                objects[i] = proposals[picked[i]];
            }
            //后处理end
            last_t = this_t;
            this_t = chrono::system_clock::now();
            std::cout << "each msg:"<<std::chrono::duration_cast<std::chrono::milliseconds>(this_t - last_t).count() << "ms;pure delay:"<<std::chrono::duration_cast<std::chrono::milliseconds>(this_t - img_time_stamp).count() << "ms;" << std::endl;
            #ifdef __SHOWING_IMG__
            draw_img(show_img,objects);
            imshow("src",show_img);
            waitKey(1);
            #endif
			float x = 0.f, y = 0.f, z = 0.f;
			bool Detect_Success = armour_process(x, y, z);
			if (Detect_Success) {
				(*communicator).send(x, y, z, 0);
			}else {
				printf("detect not target!!!\n");
			}
        }
        else
        {
            preds_lock.unlock();
            this_thread::yield();//让出cpu时间片
        }
    }
}

void restore_obj(Object& obj, float scale_ratio) {
	obj.x1 /= scale_ratio;
	obj.y1 /= scale_ratio;
	obj.x2 /= scale_ratio;
	obj.y2 /= scale_ratio;
	obj.x3 /= scale_ratio;
	obj.y3 /= scale_ratio;
	obj.x4 /= scale_ratio;
	obj.y4 /= scale_ratio;
}

bool Detector::armour_process(float& x, float& y, float& z) {
	if (objects.empty()) return false;
	int dist = INT_MAX;
	Object ans_armour;
	for (unsigned int index = 0; index < objects.size(); ++index) {
		int x_temp = (objects[index].x1 + objects[index].x2 + objects[index].x3 + objects[index].x4) / 4;
		int y_temp = (objects[index].y1 + objects[index].y2 + objects[index].y3 + objects[index].y4) / 4;
		int dist_temp = sqrt(pow(x_temp - INPUT_W / 2, 2) + pow(y_temp - INPUT_H / 2, 2));
		if (dist_temp < dist) {
			ans_armour = objects[index];
			dist = dist_temp;
		}
	}
	restore_obj(ans_armour, scale_ratio);
	Marker marker(ans_armour);
	float marker_width = get_armour_len(marker);
    //printf("marker_width:%f\n", marker_width);
	float focal_length = (params.camera_fx + params.camera_fy) * 0.5f;
	float real_L = params.transformer_small_marker_size; // 小装甲板
	float depth = (real_L * focal_length) / (marker_width);
	z = depth;
	x = (marker.center.x - params.camera_cx) / params.camera_fx * z;
	y = (marker.center.y - params.camera_cy) / params.camera_fy * z;
    printf("x=%fmm, y=%fmm, z=%fmm\n",x*1000,y*1000,z*1000);
#ifdef __ANTI_TOP__
	if (setting->anti_top_mode == 1) {
		anti_top.anti_top_run(x, z, setting->shoot_time);
		car_center_x = anti_top.center_x;
	}
	else {
		anti_top.anti_top_clear();
		anti_top.last_rotate_angle = anti_top.cur_rotate_angle;
	}
#endif
	return true;
}

float Detector::get_armour_len(Marker& marker) {
	Point2f c2c = marker.LEDs[0].center - marker.LEDs[1].center;
	auto mean_LEDs_angle = 0.5f*(marker.LEDs[0].dir_angle + marker.LEDs[1].dir_angle);
	float dir_angle_temp = params.armour_dir_angle;
	auto temp = tan(mean_LEDs_angle) * cos(dir_angle_temp + setting->pitch_angle) / sin(dir_angle_temp);  //中间变量
	float top_rotate_angle = 0.f;
	if (temp >= 0.9 || temp <= -0.9) {
		top_rotate_angle = anti_top.cur_rotate_angle - anti_top.angle_speed * 0.02;
	}
	else {
		top_rotate_angle = asin(temp);
	}
	anti_top.cur_rotate_angle = top_rotate_angle;
	float marker_width = (float)cv::norm(c2c);
	marker_width /= cos(top_rotate_angle);
	return marker_width;
}

//common functions
void Detector::generate_grids_and_stride(const std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_y = INPUT_H / stride;
        int num_grid_x = INPUT_W / stride;
        for (int gy = 0; gy < num_grid_y; gy++)
        {
            for (int gx = 0; gx < num_grid_x; gx++)
            {
                grid_strides.push_back((GridAndStride){gx, gy, stride});
            }
        }
    }
}

float* Detector::blobFromImage(cv::Mat& img, float* blob){
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (int c = 0; c < channels; c++)
    {
        for (int  h = 0; h < img_h; h++)
        {
            for (int w = 0; w < img_w; w++)
            {
                blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return blob;
}

void Detector::qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].conf;

    while (i <= j)
    {
        while (objects[i].conf > p)
            i++;

        while (objects[j].conf < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}

void Detector::qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;
    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

bool Detector::is_overlap(const Object& a, const Object& b)
{
    //static cv::Rect2f boxa, boxb;
    static float b1x1,b1y1,b1x2,b1y2,b2x1,b2y1,b2x2,b2y2;
    // boxa.x = std::min({a.x1,a.x2,a.x3,a.x4});
    // boxa.y = std::min({a.y1,a.y2,a.y3,a.y4});
    // boxa.width = std::max({a.x1,a.x2,a.x3,a.x4})-boxa.x;
    // boxa.height = std::max({a.y1,a.y2,a.y3,a.y4})-boxa.y;
    // boxb.x = std::min({b.x1,b.x2,b.x3,b.x4});
    // boxb.y = std::min({b.y1,b.y2,b.y3,b.y4});
    // boxb.width = std::max({b.x1,b.x2,b.x3,b.x4})-boxb.x;
    // boxb.height = std::max({b.y1,b.y2,b.y3,b.y4})-boxb.y;
    b1x1 = std::min({a.x1,a.x2,a.x3,a.x4});
    b1y1 = std::min({a.y1,a.y2,a.y3,a.y4});
    b1x2 = std::max({a.x1,a.x2,a.x3,a.x4});
    b1y2 = std::max({a.y1,a.y2,a.y3,a.y4});
    b2x1 = std::min({b.x1,b.x2,b.x3,b.x4});
    b2y1 = std::min({b.y1,b.y2,b.y3,b.y4});
    b2x2 = std::max({b.x1,b.x2,b.x3,b.x4});
    b2y2 = std::max({b.y1,b.y2,b.y3,b.y4});
    //return (boxa & boxb).area() > 0;
    return std::max({0.0f, std::min({b1x2,b2x2})-std::max({b1x1,b2x1})}) * std::max({0.0f, std::min({b1y2,b2y2})-std::max({b1y1,b2y1})}) > 0;
}


void Detector::generate_proposals(std::vector<GridAndStride>& grid_strides, float* preds, float prob_threshold, std::vector<Object>& proposals)
{
    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int gridx = grid_strides[anchor_idx].gridx;
        const int gridy = grid_strides[anchor_idx].gridy;
        const int stride = grid_strides[anchor_idx].stride;
        const int basic_pos = anchor_idx * (NUM_CLASSES + 9);

        //decode logic
        float x1 = (preds[basic_pos+0] + gridx) * stride;
        float y1 = (preds[basic_pos+1] + gridy) * stride;
        float x2 = (preds[basic_pos+2] + gridx) * stride;
        float y2 = (preds[basic_pos+3] + gridy) * stride;
        float x3 = (preds[basic_pos+4] + gridx) * stride;
        float y3 = (preds[basic_pos+5] + gridy) * stride;
        float x4 = (preds[basic_pos+6] + gridx) * stride;
        float y4 = (preds[basic_pos+7] + gridy) * stride;

        //choose class with highest probability
        float objectness = preds[basic_pos+8];
        int label = -1;
        float conf = 0;
        for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
        {
            float cls_score = preds[basic_pos + 9 + class_idx];
            float cls_conf = objectness * cls_score;
            if (cls_conf > prob_threshold && cls_conf>conf)
            {
                conf = cls_conf;
                label = class_idx;
            }
        } // class loop

        //save selected objects
        if(label>-1)
        {
            Object obj;
            obj.x1 = x1;
            obj.y1 = y1;
            obj.x2 = x2;
            obj.y2 = y2;
            obj.x3 = x3;
            obj.y3 = y3;
            obj.x4 = x4;
            obj.y4 = y4;
            obj.label = label;
            obj.conf = conf;
            proposals.push_back(obj);
        }
    } // point anchor loop
}


void Detector::nms(const std::vector<Object>& proposals, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = proposals.size();

    for (int i = 0; i < n; i++)
    {
        const Object& a = proposals[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = proposals[picked[j]];
            if(b.label==a.label&&is_overlap(a,b))
            {
                keep=0;
            }
        }
        if (keep)
            picked.push_back(i);
    }
}

    


void Detector::draw_img(cv::Mat& img,std::vector<Object>& objects)
{
    static cv::Point2i p1,p2,p3,p4;
    int n = objects.size();
    for(int i=0; i<n; i++)
    {
        p1 = cv::Point2i(objects[i].x1,objects[i].y1);
        p2 = cv::Point2i(objects[i].x2,objects[i].y2);
        p3 = cv::Point2i(objects[i].x3,objects[i].y3);
        p4 = cv::Point2i(objects[i].x4,objects[i].y4);
//            p1 = cv::Point2i(objects[i].x1/scale_ratio,objects[i].y1/scale_ratio);
//            p2 = cv::Point2i(objects[i].x2/scale_ratio,objects[i].y2/scale_ratio);
//            p3 = cv::Point2i(objects[i].x3/scale_ratio,objects[i].y3/scale_ratio);
//            p4 = cv::Point2i(objects[i].x4/scale_ratio,objects[i].y4/scale_ratio);
        cv::line(img, p1, p2,cv::Scalar(255,0,0));
        cv::line(img, p2, p3,cv::Scalar(0,255,0));
        cv::line(img, p3, p4,cv::Scalar(0,0,255));
        cv::line(img, p4, p1,cv::Scalar(255,255,255));

        char text[256];
        sprintf(text, "%d %.1f%%", objects[i].label, objects[i].conf * 100);
        cv::putText(img, text, p1,cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255) , 1);
    }

}



