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
	if (colors.empty()) return cv::Vec3b{ 0, 0, 0 };
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

static void pixelsInQuad(
	const std::array<cv::Point2f, 4>& quad,
	const cv::Mat& image,
	std::function<void(int x, int y)> callback)
{
	// ---- 1. Compute bounding box ----
	float minX = quad[0].x, maxX = quad[0].x;
	float minY = quad[0].y, maxY = quad[0].y;

	for (int i = 1; i < 4; ++i) {
		minX = std::min(minX, quad[i].x);
		maxX = std::max(maxX, quad[i].x);
		minY = std::min(minY, quad[i].y);
		maxY = std::max(maxY, quad[i].y);
	}

	// Clamp to image boundaries
	int x0 = std::max(0, (int)std::floor(minX));
	int x1 = std::min(image.cols - 1, (int)std::ceil(maxX));
	int y0 = std::max(0, (int)std::floor(minY));
	int y1 = std::min(image.rows - 1, (int)std::ceil(maxY));

	// ---- 2. Prepare polygon for pointPolygonTest ----
	std::vector<cv::Point2f> polygon(quad.begin(), quad.end());

	// ---- 3. Loop through bounding box ----
	for (int y = y0; y <= y1; ++y) {
		for (int x = x0; x <= x1; ++x) {

			// Test pixel center
			cv::Point2f p(x + 0.5f, y + 0.5f);

			// > 0 = inside, =0 = on edge, <0 = outside
			if (cv::pointPolygonTest(polygon, p, false) >= 0) {
				callback(x, y);
			}
		}
	}
}

static std::array<int, 2> lookupMaskCoordinate(int x, int y, const cv::Mat& H)
{
	std::vector<cv::Point2f> srcPnt{ cv::Point2f(x, y) };
	std::vector<cv::Point2f> dstPnt{};
	cv::perspectiveTransform(srcPnt, dstPnt, H);
	return std::array<int, 2>{(int)roundf(dstPnt[0].x), (int)roundf(dstPnt[0].y)};
}

int main()
{
	std::string video_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\reception\\calibrationandtext.mkv";
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

	cv::Mat H = (cv::Mat_<double>(3, 3) <<
		0.9598, 0.0150, 37.9037,
		-0.0320, 0.9926, 46.5147,
		-0.0000, 0.0000, 1.0000
		);
	cv::Mat H_inv = H.inv();

	// order topleft, topright, bottomleft, bottomright
	std::vector<std::array<cv::Point2f, 4>> transformed_boxes{};
	// translate boxes
	for (const auto& box : boxes) {
		std::vector<cv::Point2f> srcPnts{};
		std::vector<cv::Point2f> dstPnts{};

		srcPnts.push_back(cv::Point2f(box.x, box.y));
		srcPnts.push_back(cv::Point2f(box.x + box.w, box.y));
		srcPnts.push_back(cv::Point2f(box.x, box.y + box.h));
		srcPnts.push_back(cv::Point2f(box.x + box.w, box.y + box.h));
		cv::perspectiveTransform(srcPnts, dstPnts, H);
		transformed_boxes.push_back(std::array<cv::Point2f, 4>{
			dstPnts[0], dstPnts[1], dstPnts[2], dstPnts[3]
		});
	}

	// translate mask

	constexpr int START_FRAME{ 4499 };
	constexpr int FRAME_COUNT = 29;

	std::vector<cv::Vec3b> text_colors{};

	cv::Mat frame;
	for (int i = 0; i < FRAME_COUNT; ++i) {
		cap.set(cv::CAP_PROP_POS_FRAMES, START_FRAME + (i * 24));
		bool ret = cap.read(frame);
		if (!ret) {
			throw std::runtime_error("Failed to read frame");
		}

		for (const auto& box : transformed_boxes) {
			std::vector<cv::Vec3b> colors{};
			pixelsInQuad(box, frame, [&](int x, int y) {
				auto coords = lookupMaskCoordinate(x, y, H_inv);
				//std::cout << "test\n";
				if (mask.at<cv::Vec3b>(coords[1], coords[0])[1] == 255) {
					colors.push_back(frame.at<cv::Vec3b>(y, x));
				}
				});

			// compute average
			text_colors.push_back(averageColor(colors));
		}
		//cv::imshow("winname", frame);
		//cv::waitKey();
	}

	{
		std::ofstream csv_output("text_colors.csv");
		if (!csv_output) {
			throw std::runtime_error("Failed to create file for writing");
		}
		csv_output << "r,g,b\n";
		for (const auto& color : text_colors) {
			csv_output << (int)color[2] << "," << (int)color[1] << "," << (int)color[0] << "\n";
		}
	}
	cv::waitKey();

	return 0;
}