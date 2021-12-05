//
// Created by wx on 18-4-28.
//

#ifndef ROBOMASTERRUNEDETECTOR_SETTING_H
#define ROBOMASTERRUNEDETECTOR_SETTING_H
class Setting {
 public:
  Setting() { mode = 0; color = 0;anti_top_mode=0;shoot_time=0;}

 public:
  float car_v=0.f; //对面装甲板的平动角速度
  float pitch_angle = 0.f; //云台仰角
  float shoot_time; //子弹飞行时间
  int anti_top_mode; //0：不开启反陀螺，1：开启反陀螺
  int mode;  // 0：not working； 1：auto-shooooot
  int color; // 0: all 1: red 2: blue
};

#endif  // ROBOMASTERRUNEDETECTOR_SETTING_H
