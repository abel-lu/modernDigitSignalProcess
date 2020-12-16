#include<stdio.h>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace  cv;

cv::Mat addSaltNoise(const Mat srcImage, int n);
double generateGaussianNoise(double mu, double sigma);
cv::Mat addGaussianNoise(Mat &srcImag);
void Kalman();
double caluMSE(cv::Mat srcImage, cv::Mat filterImage);

int main()
{
	cv::Mat srcImage = cv::imread("E://素材//形态学图片//lena.jpg");
	cv::Mat saltNoiseImage;
	cv::Mat gaussianImage;
	cv::Mat median_Mat;
	cv::Mat Guass_Mat;
	cv::Mat boxblur_Mat;
	cv::Mat blur_Mat;
	cv::Mat bilateral_Mat;
	cv::Mat median_Mat1;
	cv::Mat Guass_Mat1;
	cv::Mat blur_Mat1;
	cv::Mat boxblur_Mat1;
	cv::Mat bilateral_Mat1;

	//均方误差：滤波器评价指标
	double boxblur_MSE, blur_MSE, bilateral_MSE, median_MSE, Guass_MSE;
	double boxblur_MSE1, blur_MSE1, bilateral_MSE1, median_MSE1, Guass_MSE1;

	if (srcImage.empty())
	{
		std::cout << "image fail to open!" << endl;
		return 0;
	}
	//原图
	cv::imshow("原图", srcImage);


	//图片加噪声，椒盐和高斯噪声
	//添加椒盐噪声
	saltNoiseImage = addSaltNoise(srcImage, 5000);
	//椒盐噪声图片
	cv::imshow("椒盐噪声", saltNoiseImage);
	// cv::imwrite("E://素材//形态学图片//saltNoiseImage.jpg", saltNoiseImage);
	//图像添加高斯噪声
	gaussianImage = addGaussianNoise(srcImage);
	//高斯噪声图片
	cv::imshow("高斯噪声", gaussianImage);
	//cv::imwrite("E://素材//形态学图片//gaussianImage.jpg", gaussianImage);


	//以下为非线性滤波，中值滤波和双边滤波
	//中值滤波
	medianBlur(gaussianImage, median_Mat, 3);
	medianBlur(saltNoiseImage, median_Mat1, 3);
	cv::imshow("中值滤波（高斯噪声）", median_Mat);
	cv::imshow("中值滤波（椒盐噪声）", median_Mat1);
	median_MSE = caluMSE(srcImage, median_Mat);
	median_MSE1 = caluMSE(srcImage, median_Mat1);
	//cv::imwrite("E://素材//形态学图片//中值滤波.jpg", median_Mat);
	//双边滤波
	bilateralFilter(gaussianImage, bilateral_Mat, 50, 50 * 2, 50 / 2);
	bilateralFilter(saltNoiseImage, bilateral_Mat1, 50, 50 * 2, 50 / 2);
	cv::imshow("双边滤波（高斯噪声）", bilateral_Mat);
	cv::imshow("双边滤波（椒盐噪声）", bilateral_Mat1);
	boxblur_MSE = caluMSE(srcImage, bilateral_Mat);
	boxblur_MSE1 = caluMSE(srcImage, bilateral_Mat1);
	//cv::imwrite("E://素材//形态学图片//双边滤波.jpg", bilateral_Mat);


	//以下为线性滤波，方框滤波、均值滤波和高斯滤波
	//方框滤波
	boxFilter(gaussianImage, boxblur_Mat, -1, Size(5, 5));
	boxFilter(saltNoiseImage, boxblur_Mat1, -1, Size(5, 5));
	cv::imshow("方框滤波（高斯噪声）", boxblur_Mat);
	cv::imshow("方框滤波（椒盐噪声）", boxblur_Mat1);
	bilateral_MSE = caluMSE(srcImage, boxblur_Mat);
	bilateral_MSE1 = caluMSE(srcImage, boxblur_Mat1);
	//cv::imwrite("E://素材//形态学图片//方框滤波.jpg", boxblur_Mat);
	//均值滤波
	blur(gaussianImage, blur_Mat, Size(5, 5), Point(-1, -1));
	blur(saltNoiseImage, blur_Mat1, Size(5, 5), Point(-1, -1));
	cv::imshow("均值滤波（高斯噪声）", blur_Mat);
	cv::imshow("均值滤波（椒盐噪声）", blur_Mat1);
	blur_MSE = caluMSE(srcImage, blur_Mat);
	blur_MSE1 = caluMSE(srcImage, blur_Mat1);
	//cv::imwrite("E://素材//形态学图片//均值滤波.jpg", blur_Mat);
	//高斯滤波
	GaussianBlur(gaussianImage, Guass_Mat, Size(5, 5), 2, 0);
	GaussianBlur(saltNoiseImage, Guass_Mat1, Size(5, 5), 2, 0);
	cv::imshow("高斯滤波（高斯噪声）", Guass_Mat);
	cv::imshow("高斯滤波（椒盐噪声）", Guass_Mat1);
	Guass_MSE = caluMSE(srcImage, Guass_Mat);
	Guass_MSE1 = caluMSE(srcImage, Guass_Mat1);
	//cv::imwrite("E://素材//形态学图片//高斯滤波.jpg", Guass_Mat);
	cout << "高斯：" << Guass_MSE << "均值：" << blur_MSE << "双边：" << bilateral_MSE << "方框：" << boxblur_MSE << "中值：" << median_MSE <<
		"高斯：" << Guass_MSE1 << "均值：" << blur_MSE1 << "双边：" << bilateral_MSE1 << "、方框：" << boxblur_MSE1 << "中值：" << median_MSE1 << endl;
	//卡尔曼滤波
	Kalman();
	waitKey(0);
	return 0;
}

double caluMSE(cv::Mat srcImage, cv::Mat filterImage)
{
	int width = srcImage.cols;
	int height = srcImage.rows;
	double mse = 0;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			mse += (srcImage.at<Vec3b>(i, j)[0] - filterImage.at<Vec3b>(i, j)[0])*(srcImage.at<Vec3b>(i, j)[0] - filterImage.at<Vec3b>(i, j)[0])
				+ (srcImage.at<Vec3b>(i, j)[1] - filterImage.at<Vec3b>(i, j)[1])*(srcImage.at<Vec3b>(i, j)[1] - filterImage.at<Vec3b>(i, j)[1])
				+ (srcImage.at<Vec3b>(i, j)[2] - filterImage.at<Vec3b>(i, j)[2])*(srcImage.at<Vec3b>(i, j)[2] - filterImage.at<Vec3b>(i, j)[2]);
		}
	}
	return mse / ((width*height) * 3);
}

void Kalman()
{

}

//添加椒盐噪声
Mat addSaltNoise(const Mat srcImage, int n)
{
	Mat dstImage = srcImage.clone();
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;		//盐噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 255;
			dstImage.at<Vec3b>(i, j)[1] = 255;
			dstImage.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;		//椒噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 0;
			dstImage.at<Vec3b>(i, j)[1] = 0;
			dstImage.at<Vec3b>(i, j)[2] = 0;
		}
	}
	return dstImage;
}

// 生成高斯噪声
double generateGaussianNoise(double mu, double sigma)
{
	//定义小值
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flag为假构造高斯随机变量X
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//构造随机变量
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flag为真构造高斯随机变量
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0*sigma + mu;
}

//为图像加入高斯噪声
Mat addGaussianNoise(Mat &srcImag)
{
	Mat dstImage = srcImag.clone();
	int channels = dstImage.channels();
	int rowsNumber = dstImage.rows;
	int colsNumber = dstImage.cols*channels;
	//推断图像的连续性
	if (dstImage.isContinuous())
	{
		colsNumber *= rowsNumber;
		rowsNumber = 1;
	}
	for (int i = 0; i < rowsNumber; i++)
	{
		for (int j = 0; j < colsNumber; j++)
		{
			//加入高斯噪声
			int val = dstImage.ptr<uchar>(i)[j] +
				generateGaussianNoise(2, 0.8) * 32;
			if (val < 0)
				val = 0;
			if (val>255)
				val = 255;
			dstImage.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return dstImage;
}