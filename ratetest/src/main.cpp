#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <span>
#include <array>
#include <bit>
#include <random>

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

static cv::Vec3b getKeyColor(const std::array<cv::Point2f, 4>& box, const cv::Mat& frame, const cv::Mat& H_inv, const cv::Mat& mask)
{
	std::vector<cv::Vec3b> colors{};
	pixelsInQuad(box, frame, [&](int x, int y) {
		auto coords = lookupMaskCoordinate(x, y, H_inv);
		if (mask.at<cv::Vec3b>(coords[1], coords[0])[1] == 255) {
			colors.push_back(frame.at<cv::Vec3b>(y, x));
		}
		});
	return averageColor(colors);
}

static double dist(const cv::Vec3b& a, const cv::Vec3b& b)
{
	cv::Vec3d d = cv::Vec3d(a) - cv::Vec3d(b);
	return cv::norm(d);
}

static bool isWhite(const cv::Vec3b& measured,
	const cv::Vec3b& white,
	const cv::Vec3b& black)
{
	return dist(measured, white) < dist(measured, black);
}

static std::vector<bool> getPRBS7(int num_bits, uint32_t seed)
{
	std::vector<bool> out{};

	std::mt19937 rng(seed);
	std::bernoulli_distribution bit;

	for (int i = 0; i < num_bits; ++i)
	{
		out.push_back(bit(rng));
	}

	return out;
}

cv::Vec3b lerpGreenRed(float t)
{
	t = std::clamp(t, 0.0f, 1.0f);

	cv::Vec3b green(0.0, 255, 0.0); // BGR
	cv::Vec3b red(0.0, 0.0, 255);

	return ((1.0f - t) * (cv::Vec3d)green) + (t * (cv::Vec3d)red);
}

cv::Scalar berToColor(float ber, float ber_min, float ber_max)
{

	ber = std::clamp(ber, ber_min, ber_max);

	// Log-scale normalize
	float t = (std::log10(ber) - std::log10(ber_min)) /
		(std::log10(ber_max) - std::log10(ber_min));

	// Hue: green (120) → red (0)
	float hue = (1.0f - t) * 120.0f;

	// OpenCV HSV: H = [0,180], S,V = [0,255]
	cv::Mat hsv(1, 1, CV_8UC3,
		cv::Scalar(hue / 2.0f, 255, 255));

	cv::Mat bgr;
	cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

	return bgr.at<cv::Vec3b>(0, 0);
}

int main()
{
	std::string video_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\new\\freqsweep.mkv";
	std::string bboxes_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\bboxes.csv";
	std::string mask_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\reception\\mask2.png";

	// start frame 759
	constexpr int START_FRAME = 721;
	constexpr int FRAMES = 500;
	constexpr int FRAME_GAP = 12; // 24 for 10hz, 12 for 20hz
	constexpr int WHITE_FRAME = 673;
	constexpr int BLACK_FRAME = 444;

	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) {
		throw std::runtime_error("Error: Could not open video");
	}

	cv::Mat mask = cv::imread(mask_path);
	if (mask.empty()) {
		throw std::runtime_error("Failed to read mask image");
	}

	auto boxes = loadCsvBoxes(bboxes_path);

	cv::Mat H{};
	std::array<cv::Point2f, 4> srcPnts{};
	srcPnts[0] = cv::Point2f(0, 0);
	srcPnts[1] = cv::Point2f(1919, 0);
	srcPnts[2] = cv::Point2f(0, 1079);
	srcPnts[3] = cv::Point2f(1919, 1079);
	std::array<cv::Point2f, 4> dstPnts{};
	dstPnts[0] = cv::Point2f(14.8, 47.9);
	dstPnts[1] = cv::Point2f(2002.2, -8.1);
	dstPnts[2] = cv::Point2f(18.4, 1155.4);
	dstPnts[3] = cv::Point2f(2044.2, 1135.4);
	H = cv::findHomography(srcPnts, dstPnts);
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

	// establish white (on) values
	std::vector<cv::Vec3b> white_values{};
	{
		cap.set(cv::CAP_PROP_POS_FRAMES, WHITE_FRAME);
		cv::Mat frame;
		bool ret = cap.read(frame);
		if (!ret) {
			throw std::runtime_error("Failed to read frame");
		}
		for (const auto& box : transformed_boxes) {
			white_values.push_back(getKeyColor(box, frame, H_inv, mask));
		}
	}
	// establish black (off) values
	std::vector<cv::Vec3b> black_values{};
	{
		cap.set(cv::CAP_PROP_POS_FRAMES, BLACK_FRAME);
		cv::Mat frame;
		bool ret = cap.read(frame);
		if (!ret) {
			throw std::runtime_error("Failed to read frame");
		}
		for (const auto& box : transformed_boxes) {
			black_values.push_back(getKeyColor(box, frame, H_inv, mask));
		}
	}

	std::array<std::vector<bool>, 109> original_bits{};
	{
		for (int i = 0; i < 109; ++i) {
			original_bits[i] = getPRBS7(500, i);
		}
	}
	std::array<std::vector<bool>, 109> received_bits;

	{
		cv::Mat frame;
		for (int i = 0; i < FRAMES; ++i) {
			printf("Frame %d/%d\n", i + 1, FRAMES);
			cap.set(cv::CAP_PROP_POS_FRAMES, START_FRAME + (i * FRAME_GAP));
			{
				bool ret = cap.read(frame);
				if (!ret) {
					throw std::runtime_error("Failed to read frame");
				}
			}

			int key_idx = 0;
			for (const auto& box : transformed_boxes) {
				auto color = getKeyColor(box, frame, H_inv, mask);
				received_bits[key_idx].push_back(isWhite(color, white_values[key_idx], black_values[key_idx]));
				++key_idx;
			}
		}
	}

	// display calibration data

	//{
	//	cv::Mat out(1080, 1920, CV_8UC3);
	//	for (int x = 0; x < 1920; ++x) {
	//		for (int y = 0; y < 1080; ++y) {
	//			auto& ref = out.at<cv::Vec3b>(y, x);
	//			ref[0] = 0;
	//			ref[1] = 0;
	//			ref[2] = 0;
	//		}
	//	}

	//	int i = 0;
	//	for (const auto& box : transformed_boxes) {
	//		cv::Scalar color = white_values[i];
	//		cv::line(out, box[0], box[1], color, 5);
	//		cv::line(out, box[1], box[3], color, 5);
	//		cv::line(out, box[3], box[2], color, 5);
	//		cv::line(out, box[2], box[0], color, 5);
	//		++i;
	//	}
	//	cv::imshow("out", out);
	//	cv::waitKey();
	//}

	//{
	//	cv::Mat out(1080, 1920, CV_8UC3);
	//	for (int x = 0; x < 1920; ++x) {
	//		for (int y = 0; y < 1080; ++y) {
	//			auto& ref = out.at<cv::Vec3b>(y, x);
	//			ref[0] = 0;
	//			ref[1] = 0;
	//			ref[2] = 0;
	//		}
	//	}

	//	int i = 0;
	//	for (const auto& box : transformed_boxes) {
	//		cv::Scalar color = black_values[i];
	//		cv::line(out, box[0], box[1], color, 5);
	//		cv::line(out, box[1], box[3], color, 5);
	//		cv::line(out, box[3], box[2], color, 5);
	//		cv::line(out, box[2], box[0], color, 5);
	//		++i;
	//	}
	//	cv::imshow("out", out);
	//	cv::waitKey();
	//}

	std::array<int, 109> errors_per_key{};
	for (int key_idx = 0; key_idx < 109; ++key_idx) {
		int num_errors = 0;
		for (int i = 0; i < 500; ++i) {
			bool original_bit = original_bits[key_idx][i];
			bool received_bit = received_bits[key_idx][i];
			if (original_bit != received_bit) {
				++num_errors;
			}
		}
		errors_per_key[key_idx] = num_errors;
	}

	// most errors
	int max_errors = 0;
	for (int errors : errors_per_key) {
		if (errors > max_errors) {
			max_errors = errors;
		}
	}

	{
		cv::Mat out(1080, 1920, CV_8UC3);
		for (int x = 0; x < 1920; ++x) {
			for (int y = 0; y < 1080; ++y) {
				auto& ref = out.at<cv::Vec3b>(y, x);
				ref[0] = 0;
				ref[1] = 0;
				ref[2] = 0;
			}
		}

		int i = 0;
		for (const auto& box : boxes) {
			int errors = errors_per_key[i];
			double rate = double(errors) / double(max_errors);
			printf("error rate: %f%%\n", rate * 100.0);
			cv::Scalar color = berToColor(rate, 1e-3, 1.0);
			cv::rectangle(out, cv::Point(box.x, box.y), cv::Point(box.x + box.w, box.y + box.h), color, cv::FILLED);
			++i;
		}
		cv::imshow("out", out);
		cv::waitKey();
	}

	return 0;
}