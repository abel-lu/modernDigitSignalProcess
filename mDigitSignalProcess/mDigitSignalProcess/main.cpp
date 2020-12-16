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
	cv::Mat srcImage = cv::imread("E://�ز�//��̬ѧͼƬ//lena.jpg");
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

	//�������˲�������ָ��
	double boxblur_MSE, blur_MSE, bilateral_MSE, median_MSE, Guass_MSE;
	double boxblur_MSE1, blur_MSE1, bilateral_MSE1, median_MSE1, Guass_MSE1;

	if (srcImage.empty())
	{
		std::cout << "image fail to open!" << endl;
		return 0;
	}
	//ԭͼ
	cv::imshow("ԭͼ", srcImage);


	//ͼƬ�����������κ͸�˹����
	//��ӽ�������
	saltNoiseImage = addSaltNoise(srcImage, 5000);
	//��������ͼƬ
	cv::imshow("��������", saltNoiseImage);
	// cv::imwrite("E://�ز�//��̬ѧͼƬ//saltNoiseImage.jpg", saltNoiseImage);
	//ͼ����Ӹ�˹����
	gaussianImage = addGaussianNoise(srcImage);
	//��˹����ͼƬ
	cv::imshow("��˹����", gaussianImage);
	//cv::imwrite("E://�ز�//��̬ѧͼƬ//gaussianImage.jpg", gaussianImage);


	//����Ϊ�������˲�����ֵ�˲���˫���˲�
	//��ֵ�˲�
	medianBlur(gaussianImage, median_Mat, 3);
	medianBlur(saltNoiseImage, median_Mat1, 3);
	cv::imshow("��ֵ�˲�����˹������", median_Mat);
	cv::imshow("��ֵ�˲�������������", median_Mat1);
	median_MSE = caluMSE(srcImage, median_Mat);
	median_MSE1 = caluMSE(srcImage, median_Mat1);
	//cv::imwrite("E://�ز�//��̬ѧͼƬ//��ֵ�˲�.jpg", median_Mat);
	//˫���˲�
	bilateralFilter(gaussianImage, bilateral_Mat, 50, 50 * 2, 50 / 2);
	bilateralFilter(saltNoiseImage, bilateral_Mat1, 50, 50 * 2, 50 / 2);
	cv::imshow("˫���˲�����˹������", bilateral_Mat);
	cv::imshow("˫���˲�������������", bilateral_Mat1);
	boxblur_MSE = caluMSE(srcImage, bilateral_Mat);
	boxblur_MSE1 = caluMSE(srcImage, bilateral_Mat1);
	//cv::imwrite("E://�ز�//��̬ѧͼƬ//˫���˲�.jpg", bilateral_Mat);


	//����Ϊ�����˲��������˲�����ֵ�˲��͸�˹�˲�
	//�����˲�
	boxFilter(gaussianImage, boxblur_Mat, -1, Size(5, 5));
	boxFilter(saltNoiseImage, boxblur_Mat1, -1, Size(5, 5));
	cv::imshow("�����˲�����˹������", boxblur_Mat);
	cv::imshow("�����˲�������������", boxblur_Mat1);
	bilateral_MSE = caluMSE(srcImage, boxblur_Mat);
	bilateral_MSE1 = caluMSE(srcImage, boxblur_Mat1);
	//cv::imwrite("E://�ز�//��̬ѧͼƬ//�����˲�.jpg", boxblur_Mat);
	//��ֵ�˲�
	blur(gaussianImage, blur_Mat, Size(5, 5), Point(-1, -1));
	blur(saltNoiseImage, blur_Mat1, Size(5, 5), Point(-1, -1));
	cv::imshow("��ֵ�˲�����˹������", blur_Mat);
	cv::imshow("��ֵ�˲�������������", blur_Mat1);
	blur_MSE = caluMSE(srcImage, blur_Mat);
	blur_MSE1 = caluMSE(srcImage, blur_Mat1);
	//cv::imwrite("E://�ز�//��̬ѧͼƬ//��ֵ�˲�.jpg", blur_Mat);
	//��˹�˲�
	GaussianBlur(gaussianImage, Guass_Mat, Size(5, 5), 2, 0);
	GaussianBlur(saltNoiseImage, Guass_Mat1, Size(5, 5), 2, 0);
	cv::imshow("��˹�˲�����˹������", Guass_Mat);
	cv::imshow("��˹�˲�������������", Guass_Mat1);
	Guass_MSE = caluMSE(srcImage, Guass_Mat);
	Guass_MSE1 = caluMSE(srcImage, Guass_Mat1);
	//cv::imwrite("E://�ز�//��̬ѧͼƬ//��˹�˲�.jpg", Guass_Mat);
	cout << "��˹��" << Guass_MSE << "��ֵ��" << blur_MSE << "˫�ߣ�" << bilateral_MSE << "����" << boxblur_MSE << "��ֵ��" << median_MSE <<
		"��˹��" << Guass_MSE1 << "��ֵ��" << blur_MSE1 << "˫�ߣ�" << bilateral_MSE1 << "������" << boxblur_MSE1 << "��ֵ��" << median_MSE1 << endl;
	//�������˲�
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

//��ӽ�������
Mat addSaltNoise(const Mat srcImage, int n)
{
	Mat dstImage = srcImage.clone();
	for (int k = 0; k < n; k++)
	{
		//���ȡֵ����
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//ͼ��ͨ���ж�
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;		//������
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
		//���ȡֵ����
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//ͼ��ͨ���ж�
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;		//������
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

// ���ɸ�˹����
double generateGaussianNoise(double mu, double sigma)
{
	//����Сֵ
	const double epsilon = numeric_limits<double>::min();
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flagΪ�ٹ����˹�������X
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//�����������
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flagΪ�湹���˹�������
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0*sigma + mu;
}

//Ϊͼ������˹����
Mat addGaussianNoise(Mat &srcImag)
{
	Mat dstImage = srcImag.clone();
	int channels = dstImage.channels();
	int rowsNumber = dstImage.rows;
	int colsNumber = dstImage.cols*channels;
	//�ƶ�ͼ���������
	if (dstImage.isContinuous())
	{
		colsNumber *= rowsNumber;
		rowsNumber = 1;
	}
	for (int i = 0; i < rowsNumber; i++)
	{
		for (int j = 0; j < colsNumber; j++)
		{
			//�����˹����
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