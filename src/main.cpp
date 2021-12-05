#include <thread>
#include <chrono>
#include "Detector.hpp"
#include "MVCamera.h"
#include <opencv2/opencv.hpp>
#include "RemoteControl.h"
#include "setting.h"
#include "uart.h"

int main(int argc, char** argv) 
{
	Setting setting;
	UART commun("/dev/usbtottl", "/dev/ttyUSB0");
	Detector detector(&setting, &commun);
	RemoteControl remoteProcess(&setting, commun);
	thread receive_data(&RemoteControl::Receiver, &remoteProcess);
    thread pre(&Detector::pre_process,&detector);
    thread infer(&Detector::do_inference,&detector);
    thread post(&Detector::post_process,&detector);

    pre.join();

    return 0;
}