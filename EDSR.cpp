#include <iostream>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace dnn_superres;

Mat upscaleImage(Mat img, string modelName, string modelPath, int scale){
	DnnSuperResImpl sr;
	sr.readModel(modelPath);
	sr.setModel(modelName,scale);
	// Output image
	Mat outputImage;
	sr.upsample(img, outputImage);
	return outputImage;
}

int main(int argc, char *argv[])
{
	// Read image
	Mat img = imread("AI-Courses-By-OpenCV-Github.jpg");
	
	// Region to crop
	Rect roi;
	roi.x = 850;
	roi.y = 0;
	roi.width = img.size().width - 850;
	roi.height = 80;
	img = img(roi);

	// EDSR (x4)
	string path = "EDSR_x4.pb";
	string modelName = "edsr";
	int scale = 4;
	Mat result = upscaleImage(img, modelName, path, scale);

	// Image resized using OpenCV
	Mat resized;
	cv::resize(img, resized, cv::Size(), scale, scale);
   

	/*imshow("Original image",img);
	imshow("SR upscaled",result);
	imshow("OpenCV upscaled",resized);
	waitKey(0);
	destroyAllWindows();*/

	return 0;
}