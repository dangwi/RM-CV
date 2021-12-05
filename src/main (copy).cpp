#include <thread>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "MVCamera.h"
#include "TRTModule.hpp"

static const int IMG_W = 1280;
static const int IMG_H = 1000;
static const int INPUT_W = 640;
static const int INPUT_H = 512;
static const int NUM_CLASSES = 36;
static const float CONF_THRESH = 0.7;
static const float NMS_THRESH = 0.45;
static const std::vector<int> strides = {8, 16, 32};
float r = 1.0;

std::vector<GridAndStride> grid_strides;

inline void start_camera() {
  MVCamera::Init();
  MVCamera::Play();
  MVCamera::SetExposureTime(false, 10000);
  MVCamera::SetLargeResolution(true);
}


cv::Mat static_resize(cv::Mat& img) {
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

float* blobFromImage(cv::Mat& img, float* blob){
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return blob;
}


static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
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

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static bool is_overlap(const Object& a, const Object& b)
{
    static cv::Rect2f boxa, boxb;
    boxa.x = std::min({a.x1,a.x2,a.x3,a.x4});
    boxa.y = std::min({a.y1,a.y2,a.y3,a.y4});
    boxa.width = std::max({a.x1,a.x2,a.x3,a.x4})-boxa.x;
    boxa.height = std::max({a.y1,a.y2,a.y3,a.y4})-boxa.y;
    boxb.x = std::min({a.x1,a.x2,a.x3,a.x4});
    boxb.y = std::min({a.y1,a.y2,a.y3,a.y4});
    boxb.width = std::max({a.x1,a.x2,a.x3,a.x4})-boxb.x;
    boxb.height = std::max({a.y1,a.y2,a.y3,a.y4})-boxb.y;
    return (boxa & boxb).area() > 0;
}

//calculate offset
static void generate_grids_and_stride(const std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
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

static void generate_proposals(std::vector<GridAndStride>& grid_strides, float* preds, float prob_threshold, std::vector<Object>& proposals)
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


static void nms(const std::vector<Object>& proposals, std::vector<int>& picked, float nms_threshold)
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

static void post_process(float* preds, std::vector<Object>& objects) {
        std::vector<Object> proposals;

        generate_proposals(grid_strides, preds,  CONF_THRESH, proposals);//add offset to points(decode network outputs)
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
}

static void draw_img(cv::Mat& img,std::vector<Object>& objects)
{
    static cv::Point2i p1,p2,p3,p4;
    int n = objects.size();
    for(int i=0; i<n; i++)
    {
        p1 = cv::Point2i(objects[i].x1/r,objects[i].y1/r);
        p2 = cv::Point2i(objects[i].x2/r,objects[i].y2/r);
        p3 = cv::Point2i(objects[i].x3/r,objects[i].y3/r);
        p4 = cv::Point2i(objects[i].x4/r,objects[i].y4/r);
        cv::line(img, p1, p2,cv::Scalar(255,0,0));
        cv::line(img, p2, p3,cv::Scalar(0,255,0));
        cv::line(img, p3, p4,cv::Scalar(0,0,255));
        cv::line(img, p4, p1,cv::Scalar(255,255,255));

        char text[256];
        sprintf(text, "%d %.1f%%", objects[i].label, objects[i].conf * 100);
        cv::putText(img, text, p1,cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255) , 1);
    }

}

int main(int argc, char** argv) 
{
    r = std::min(INPUT_W / (IMG_W*1.0), INPUT_H / (IMG_H*1.0));

    start_camera();
    generate_grids_and_stride(strides, grid_strides);
    float* blob = new float[3 * INPUT_H * INPUT_W];

    TRTModule trt(INPUT_H,INPUT_W);

    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();

    float* preds;
    while (1){
        cv::Mat img;
        MVCamera::GetFrame(img);
        if(img.data)
        {

        start = std::chrono::system_clock::now();
        int img_w = img.cols;
        int img_h = img.rows;
        cv::Mat pr_img = static_resize(img);
        blobFromImage(pr_img, blob);

        preds = trt(blob);

        std::vector<Object> objects;
        post_process(preds,objects);
        std::cout<<"objects_num:"<<objects.size()<<std::endl;

        end = std::chrono::system_clock::now();
        std::cout << "total:"<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms;" << std::endl;

        draw_img(img,objects);//draw objects
        //cv::imshow("src",img);
        //cv::waitKey(1);
        }
        else 
        {
            std::cout << "noimg"<< endl;
        }
    }
    return 0;
}