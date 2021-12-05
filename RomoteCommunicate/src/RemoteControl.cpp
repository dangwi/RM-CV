//
// Created by wx on 18-4-28.
// Modified by flyingtiger on 19-03-04.
//

#include "RemoteControl.h"
#include <iostream>
#include <thread>
#include "uart.h"
#define PI 3.1415926f

bool RUNNING = true;

void RemoteControl::Receiver() {
    unsigned char buffer[8] = {0};
    while (RUNNING) {
        /// receive
        int num_bytes = communicator.receive(buffer);
        if (num_bytes <= 0) continue;
        unsigned char cmd1 = buffer[0];
        float value1 = float(buffer[1] * 256 + buffer[2]) * 1.0f / 1000;
        unsigned char cmd2 = buffer[3];
        short value2_temp = buffer[4]<<8 | buffer[5];
        short value3_temp = buffer[6]<<8 | buffer[7];
        float value2 = static_cast<float>(value2_temp)/100.f*PI/180.f;
        float value3 = static_cast<float>(value3_temp)*PI/(180.f*100.f);  //对面运动的角速度，单位：度 / 秒
        //std::cout<<"对面运动角速度!!!!!!!!!!!!!："<<value3/PI*180<<std::endl;
        //static float value3_temp1 = 0;
        //value3_temp1 = 0.8*value3_temp1+0.2*value3;
        //value3 = value3_temp1;

       // std::cout<<"对面运动角速度："<<value3/PI*180<<std::endl;

        switch (cmd1) {
            case 0x00:{
                setting->anti_top_mode = 0;
                break;
            }
            case 0x01:{
                setting->anti_top_mode = 1;
                setting->shoot_time = value1;
                break;
            }
            case 0x02:{
                if(setting->mode != 2){
                    setting->mode = 2;
                    std::cout << "Change mode 222222222222222222222222222222" << std::endl;
                }
                break;
            }
            case 0x03:{
                if(setting->mode != 3){
                    setting->mode = 3;
                    std::cout << "Change mode 333333333333333333333333333333" << std::endl;
                }
                break;
            }
            case 0x04:{
                if(setting->mode == 2||setting->mode ==3)
                    setting->mode = 4;
                break;
            }
            case 0x05:{
                setting->anti_top_mode = 2;
                break;
            }
            case 0x5f: {
                printf("-------------------------------------Yaw speed!!! %f\n", value1);
                break;
            }
            case 0x8f: {
                printf("-------------------------------------Pitch speed!!! %f\n", value1);
                break;
            }
            case 0x7f: {
                printf("-------------------------------------shutdown!!!\n");
                system("sudo shutdown now");
            }
            default:
                break;
        }

        switch (cmd2) {
            case 0x6f: {
                printf("-------------------------------------Yaw angle!!! %f\n", value2);
                break;
            }
            case 0x9f: {
                setting->pitch_angle = value2;
                setting->car_v = value3;
                break;
            }
            default:
                break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}
