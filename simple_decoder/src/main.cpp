#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <span>
#include <array>

#include <opencv2/opencv.hpp>

struct Box {
	int x;
	int y;
	int w;
	int h;
};

static std::vector<Box> loadCsvBoxes(const std::string& filename) {
	std::vector<Box> boxes;
	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open file: " + filename);
	}

	std::string line;

	// Skip header line
	if (!std::getline(file, line)) {
		return boxes; // empty file
	}

	// Parse data lines
	while (std::getline(file, line)) {
		if (line.empty()) continue;

		std::stringstream ss(line);
		std::string token;
		Box b;

		// Extract 4 integer columns
		if (std::getline(ss, token, ',')) b.x = std::stoi(token);
		if (std::getline(ss, token, ',')) b.y = std::stoi(token);
		if (std::getline(ss, token, ',')) b.w = std::stoi(token);
		if (std::getline(ss, token, ',')) b.h = std::stoi(token);

		boxes.push_back(b);
	}

	return boxes;
}

static cv::Vec3b averageColor(std::span<const cv::Vec3b> colors) {
	cv::Vec3i sum{};
	for (const auto& col : colors) {
		sum[0] += col[0];
		sum[1] += col[1];
		sum[2] += col[2];
	}
	cv::Vec3b res{};
	res[0] = static_cast<uchar>((sum[0] + colors.size() / 2) / colors.size());
	res[1] = static_cast<uchar>((sum[1] + colors.size() / 2) / colors.size());
	res[2] = static_cast<uchar>((sum[2] + colors.size() / 2) / colors.size());
	return res;
}

static auto saveVectorAsImage(const std::vector<cv::Vec3b>& pixels, int width, int height, const std::string& filename) {
	if (pixels.size() != width * height) {
		throw std::runtime_error("Pixel vector size does not match width * height");
	}

	// Create an empty image
	cv::Mat img(height, width, CV_8UC3);

	// Copy pixels into the Mat, converting from x,y order to row-major (y,x)
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int index = x + y * width; // index in x,y order
			img.at<cv::Vec3b>(y, x) = pixels[index];
		}
	}

	// Save the image
	cv::imwrite(filename, img);
	return img;
}

int main()
{
	std::string video_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\reception\\mandrill_conv.mkv";
	std::string bboxes_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\bboxes.csv";
	std::string mask_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\reception\\mask2.png";

	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) {
		throw std::runtime_error("Error: Could not open video");
	}

	cv::Mat mask = cv::imread(mask_path);
	if (mask.empty()) {
		throw std::runtime_error("Failed to read mask image");
	}

	auto boxes = loadCsvBoxes(bboxes_path);

	std::array<int, 1> START_FRAMES{ 562 };
	constexpr int FRAME_COUNT = 151;

	std::vector<cv::Vec3b> bitmap_pixels{};

	cv::Mat frame;
	for (int START_FRAME : START_FRAMES) {
		for (int i = 0; i < FRAME_COUNT; ++i) {
			cap.set(cv::CAP_PROP_POS_FRAMES, START_FRAME + (i * 24));
			bool ret = cap.read(frame);
			if (!ret) {
				throw std::runtime_error("Failed to read frame");
			}

			for (const auto& box : boxes) {
				std::vector<cv::Vec3b> colors{};
				for (int y = box.y; y < box.y + box.h; ++y) {
					for (int x = box.x; x < box.x + box.w; ++x) {
						auto test = mask.at<cv::Vec3b>(y, x);
						if (mask.at<cv::Vec3b>(y, x)[1] == 255) {
							colors.push_back(frame.at<cv::Vec3b>(y, x));
						}
					}
				}

				// compute average
				bitmap_pixels.push_back(averageColor(colors));
			}
		}
		bitmap_pixels.resize(16384);
		std::string name = std::format("testpattern{}.png", START_FRAME);
		auto img = saveVectorAsImage(bitmap_pixels, 128, 128, name);
		cv::namedWindow(name, cv::WINDOW_NORMAL);
		cv::resizeWindow(name, cv::Size{ 512, 512 });
		cv::imshow(name, img);
	}
	cv::waitKey();

	return 0;
}