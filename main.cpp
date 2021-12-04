#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct CrowdPoint
{
	cv::Point pt;
	float prob;
};

static void shift(int w, int h, int stride, vector<float> anchor_points, vector<float>& shifted_anchor_points)
{
	vector<float> x_, y_;
	for (int i = 0; i < w; i++)
	{
		float x = (i + 0.5) * stride;
		x_.push_back(x);
	}
	for (int i = 0; i < h; i++)
	{
		float y = (i + 0.5) * stride;
		y_.push_back(y);
	}

	vector<float> shift_x((size_t)w * h, 0), shift_y((size_t)w * h, 0);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			shift_x[i * w + j] = x_[j];
		}
	}
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			shift_y[i * w + j] = y_[i];
		}
	}

	vector<float> shifts((size_t)w * h * 2, 0);
	for (int i = 0; i < w * h; i++)
	{
		shifts[i * 2] = shift_x[i];
		shifts[i * 2 + 1] = shift_y[i];
	}

	shifted_anchor_points.resize((size_t)2 * w * h * anchor_points.size() / 2, 0);
	for (int i = 0; i < w * h; i++)
	{
		for (int j = 0; j < anchor_points.size() / 2; j++)
		{
			float x = anchor_points[j * 2] + shifts[i * 2];
			float y = anchor_points[j * 2 + 1] + shifts[i * 2 + 1];
			shifted_anchor_points[i * anchor_points.size() / 2 * 2 + j * 2] = x;
			shifted_anchor_points[i * anchor_points.size() / 2 * 2 + j * 2 + 1] = y;
		}
	}
}
static void generate_anchor_points(int stride, int row, int line, vector<float>& anchor_points)
{
	float row_step = (float)stride / row;
	float line_step = (float)stride / line;

	vector<float> x_, y_;
	for (int i = 1; i < line + 1; i++)
	{
		float x = (i - 0.5) * line_step - stride / 2;
		x_.push_back(x);
	}
	for (int i = 1; i < row + 1; i++)
	{
		float y = (i - 0.5) * row_step - stride / 2;
		y_.push_back(y);
	}
	vector<float> shift_x((size_t)row * line, 0), shift_y((size_t)row * line, 0);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < line; j++)
		{
			shift_x[i * line + j] = x_[j];
		}
	}
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < line; j++)
		{
			shift_y[i * line + j] = y_[i];
		}
	}
	anchor_points.resize((size_t)row * line * 2, 0);
	for (int i = 0; i < row * line; i++)
	{
		float x = shift_x[i];
		float y = shift_y[i];
		anchor_points[i * 2] = x;
		anchor_points[i * 2 + 1] = y;
	}
}
static void generate_anchor_points(int img_w, int img_h, vector<int> pyramid_levels, int row, int line, vector<float>& all_anchor_points)
{
	vector<pair<int, int> > image_shapes;
	vector<int> strides;
	for (int i = 0; i < pyramid_levels.size(); i++)
	{
		int new_h = floor((img_h + pow(2, pyramid_levels[i]) - 1) / pow(2, pyramid_levels[i]));
		int new_w = floor((img_w + pow(2, pyramid_levels[i]) - 1) / pow(2, pyramid_levels[i]));
		image_shapes.push_back(make_pair(new_w, new_h));
		strides.push_back(pow(2, pyramid_levels[i]));
	}

	all_anchor_points.clear();
	for (int i = 0; i < pyramid_levels.size(); i++)
	{
		vector<float> anchor_points;
		generate_anchor_points(pow(2, pyramid_levels[i]), row, line, anchor_points);
		vector<float> shifted_anchor_points;
		shift(image_shapes[i].first, image_shapes[i].second, strides[i], anchor_points, shifted_anchor_points);
		all_anchor_points.insert(all_anchor_points.end(), shifted_anchor_points.begin(), shifted_anchor_points.end());
	}
}

class P2PNet
{
public:
	P2PNet(const float confThreshold = 0.5)
	{
		this->confThreshold = confThreshold;
		this->net = readNet("SHTechA.onnx");
	}
	void detect(Mat& frame);
private:
	float confThreshold;
	Net net;
	Mat preprocess(Mat srcimgt);
	const float mean[3] = { 0.485, 0.456, 0.406 };
	const float std[3] = { 0.229, 0.224, 0.225 };
	vector<String> output_names = { "pred_logits", "pred_points" };
};


Mat P2PNet::preprocess(Mat srcimg)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	int new_width = srcw / 128 * 128;
	int new_height = srch / 128 * 128;
	Mat dstimg;
	cvtColor(srcimg, dstimg, cv::COLOR_BGR2RGB);
	resize(dstimg, dstimg, Size(new_width, new_height), INTER_AREA);
	dstimg.convertTo(dstimg, CV_32F);
	int i = 0, j = 0;
	for (i = 0; i < dstimg.rows; i++)
	{
		float* pdata = (float*)(dstimg.data + i * dstimg.step);
		for (j = 0; j < dstimg.cols; j++)
		{
			pdata[0] = (pdata[0] / 255.0 - this->mean[0]) / this->std[0];
			pdata[1] = (pdata[1] / 255.0 - this->mean[1]) / this->std[1];
			pdata[2] = (pdata[2] / 255.0 - this->mean[2]) / this->std[2];
			pdata += 3;
		}
	}
	return dstimg;
}

void P2PNet::detect(Mat& frame)
{
	const int width = frame.cols;
	const int height = frame.rows;
	Mat img = this->preprocess(frame);
	const int new_width = img.cols;
	const int new_height = img.rows;
	Mat blob = blobFromImage(img);
	this->net.setInput(blob);
	vector<Mat> outs;
	//this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	this->net.forward(outs, output_names);

	vector<int> pyramid_levels(1, 3);
	vector<float> all_anchor_points;
	generate_anchor_points(img.cols, img.rows, pyramid_levels, 2, 2, all_anchor_points);
	const int num_proposal = outs[0].cols;
	int i = 0;
	float* pscore = (float*)outs[0].data;
	float* pcoord = (float*)outs[1].data;
	vector<CrowdPoint> crowd_points;
	for (i = 0; i < num_proposal; i++)
	{
		if (pscore[i] > this->confThreshold)
		{
			float x = (pcoord[i] + all_anchor_points[i * 2]) / (float)new_width * (float)width;
			float y = (pcoord[i+1]+ all_anchor_points[i * 2 + 1]) / (float)new_height * (float)height;
			crowd_points.push_back({ Point(int(x), int(y)), pscore[i] });
		}
		pcoord += 2;
	}
	cout << "have " << crowd_points.size() << " people" << endl;
	for (i = 0; i < crowd_points.size(); i++)
	{
		cv::circle(frame, crowd_points[i].pt, 2, cv::Scalar(0, 0, 255), -1, 8, 0);
	}
}

int main()
{
	P2PNet mynet(0.5);
	string imgpath = "imgs/demo1.jpg";
	Mat srcimg = imread(imgpath);
	mynet.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}