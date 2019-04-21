#include "opencv2/opencv.hpp"
#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "show_cream");

    ros::NodeHandle nh;

    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("camera/image", 1);

    cv::Mat image, outImage;
    cv::VideoCapture cap;

    cap.open(1);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 960);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 540);
    ROS_INFO("Start\n");

    char ch = cv::waitKey(33);

    while (ch != 'e')
    {
        cap >> image;
        flip(image, outImage, 0);
        imshow("camera", outImage);
        sensor_msgs::ImagePtr msg =
            cv_bridge::CvImage(std_msgs::Header(), "bgr8", outImage).toImageMsg();
        pub.publish(msg);
        ch = cv::waitKey(33);
    }

    return 0;
}