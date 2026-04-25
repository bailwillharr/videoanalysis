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


static bool isOn(double val, double black, double green) {
	return std::abs(val - green) < std::abs(val - black);
}

int main()
{
	std::string video_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\new\\cells.mkv";
	std::string bboxes_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\bboxes.csv";
	std::string mask_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\reception\\mask2.png";

	constexpr int FRAMES = 100;
	constexpr double FRAME_GAP = 240.0 / 10.0;
	constexpr int START_FRAME = 8752;
	//constexpr int START_FRAME = 3468;
	constexpr int GREEN_FRAME = 350;
	constexpr int BLACK_FRAME = 460;

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

	// establish green (on) values
	std::vector<cv::Vec3b> green_values{};
	{
		cap.set(cv::CAP_PROP_POS_FRAMES, GREEN_FRAME);
		cv::Mat frame;
		bool ret = cap.read(frame);
		if (!ret) {
			throw std::runtime_error("Failed to read frame");
		}
		cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
		for (const auto& box : transformed_boxes) {
			green_values.push_back(getKeyColor(box, frame, H_inv, mask));
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
		cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
		for (const auto& box : transformed_boxes) {
			black_values.push_back(getKeyColor(box, frame, H_inv, mask));
		}
	}

	std::vector<std::vector<int>> cells;
#if 0
	{
		cells.emplace_back(std::vector<int>{ 0, });
		cells.emplace_back(std::vector<int>{ 1, });
		cells.emplace_back(std::vector<int>{ 2, });
		cells.emplace_back(std::vector<int>{ 3, });
		cells.emplace_back(std::vector<int>{ 4, });
		cells.emplace_back(std::vector<int>{ 5, 6, });
		cells.emplace_back(std::vector<int>{ 7, });
		cells.emplace_back(std::vector<int>{ 8, });
		cells.emplace_back(std::vector<int>{ 9, });
		cells.emplace_back(std::vector<int>{ 10, });
		cells.emplace_back(std::vector<int>{ 11, });
		cells.emplace_back(std::vector<int>{ 12, });
		cells.emplace_back(std::vector<int>{ 13, });
		cells.emplace_back(std::vector<int>{ 14, });
		cells.emplace_back(std::vector<int>{ 15, });
		cells.emplace_back(std::vector<int>{ 16, });
		cells.emplace_back(std::vector<int>{ 17, });
		cells.emplace_back(std::vector<int>{ 18, });
		cells.emplace_back(std::vector<int>{ 19, });
		cells.emplace_back(std::vector<int>{ 21, 20, });
		cells.emplace_back(std::vector<int>{ 22, });
		cells.emplace_back(std::vector<int>{ 23, });
		cells.emplace_back(std::vector<int>{ 24, });
		cells.emplace_back(std::vector<int>{ 25, });
		cells.emplace_back(std::vector<int>{ 26, });
		cells.emplace_back(std::vector<int>{ 27, });
		cells.emplace_back(std::vector<int>{ 28, });
		cells.emplace_back(std::vector<int>{ 29, });
		cells.emplace_back(std::vector<int>{ 30, });
		cells.emplace_back(std::vector<int>{ 31, });
		cells.emplace_back(std::vector<int>{ 32, });
		cells.emplace_back(std::vector<int>{ 33, });
		cells.emplace_back(std::vector<int>{ 34, });
		cells.emplace_back(std::vector<int>{ 35, });
		cells.emplace_back(std::vector<int>{ 36, });
		cells.emplace_back(std::vector<int>{ 37, });
		cells.emplace_back(std::vector<int>{ 38, });
		cells.emplace_back(std::vector<int>{ 39, });
		cells.emplace_back(std::vector<int>{ 40, });
		cells.emplace_back(std::vector<int>{ 41, });
		cells.emplace_back(std::vector<int>{ 42, });
		cells.emplace_back(std::vector<int>{ 43, });
		cells.emplace_back(std::vector<int>{ 44, });
		cells.emplace_back(std::vector<int>{ 45, });
		cells.emplace_back(std::vector<int>{ 46, });
		cells.emplace_back(std::vector<int>{ 47, });
		cells.emplace_back(std::vector<int>{ 48, });
		cells.emplace_back(std::vector<int>{ 49, });
		cells.emplace_back(std::vector<int>{ 51, 50, });
		cells.emplace_back(std::vector<int>{ 52, });
		cells.emplace_back(std::vector<int>{ 53, });
		cells.emplace_back(std::vector<int>{ 73, });
		cells.emplace_back(std::vector<int>{ 54, });
		cells.emplace_back(std::vector<int>{ 55, });
		cells.emplace_back(std::vector<int>{ 56, });
		cells.emplace_back(std::vector<int>{ 57, });
		cells.emplace_back(std::vector<int>{ 58, });
		cells.emplace_back(std::vector<int>{ 59, });
		cells.emplace_back(std::vector<int>{ 77, });
		cells.emplace_back(std::vector<int>{ 60, });
		cells.emplace_back(std::vector<int>{ 61, });
		cells.emplace_back(std::vector<int>{ 62, });
		cells.emplace_back(std::vector<int>{ 63, });
		cells.emplace_back(std::vector<int>{ 64, });
		cells.emplace_back(std::vector<int>{ 65, });
		cells.emplace_back(std::vector<int>{ 66, });
		cells.emplace_back(std::vector<int>{ 67, });
		cells.emplace_back(std::vector<int>{ 68, });
		cells.emplace_back(std::vector<int>{ 69, });
		cells.emplace_back(std::vector<int>{ 70, });
		cells.emplace_back(std::vector<int>{ 71, });
		cells.emplace_back(std::vector<int>{ 72, });
		cells.emplace_back(std::vector<int>{ 74, });
		cells.emplace_back(std::vector<int>{ 75, });
		cells.emplace_back(std::vector<int>{ 76, });
		cells.emplace_back(std::vector<int>{ 78, });
		cells.emplace_back(std::vector<int>{ 79, });
		cells.emplace_back(std::vector<int>{ 80, });
		cells.emplace_back(std::vector<int>{ 81, });
		cells.emplace_back(std::vector<int>{ 82, });
		cells.emplace_back(std::vector<int>{ 84, 83, });
		cells.emplace_back(std::vector<int>{ 85, });
		cells.emplace_back(std::vector<int>{ 86, });
		cells.emplace_back(std::vector<int>{ 87, });
		cells.emplace_back(std::vector<int>{ 88, });
		cells.emplace_back(std::vector<int>{ 89, });
		cells.emplace_back(std::vector<int>{ 90, });
		cells.emplace_back(std::vector<int>{ 91, });
		cells.emplace_back(std::vector<int>{ 92, });
		cells.emplace_back(std::vector<int>{ 93, });
		cells.emplace_back(std::vector<int>{ 94, });
		cells.emplace_back(std::vector<int>{ 108, });
		cells.emplace_back(std::vector<int>{ 95, });
		cells.emplace_back(std::vector<int>{ 96, });
		cells.emplace_back(std::vector<int>{ 97, });
		cells.emplace_back(std::vector<int>{ 98, });
		cells.emplace_back(std::vector<int>{ 99, });
		cells.emplace_back(std::vector<int>{ 100, });
		cells.emplace_back(std::vector<int>{ 101, });
		cells.emplace_back(std::vector<int>{ 102, });
		cells.emplace_back(std::vector<int>{ 103, });
		cells.emplace_back(std::vector<int>{ 104, });
		cells.emplace_back(std::vector<int>{ 105, });
		cells.emplace_back(std::vector<int>{ 106, });
		cells.emplace_back(std::vector<int>{ 107, });
	}
#endif
#if 0
	{
		cells.emplace_back(std::vector<int>{ 1, 21, 22, 0, 20, });
		cells.emplace_back(std::vector<int>{ 23, 24, 3, 2, });
		cells.emplace_back(std::vector<int>{ 26, 25, 4, });
		cells.emplace_back(std::vector<int>{ 5, 27, 28, 6, 7, });
		cells.emplace_back(std::vector<int>{ 29, 8, 30, });
		cells.emplace_back(std::vector<int>{ 32, 9, 31, 10, });
		cells.emplace_back(std::vector<int>{ 11, 12, 33, });
		cells.emplace_back(std::vector<int>{ 13, 35, 34, 14, });
		cells.emplace_back(std::vector<int>{ 36, 16, 37, 15, });
		cells.emplace_back(std::vector<int>{ 39, 38, 17, 18, });
		cells.emplace_back(std::vector<int>{ 40, 19, });
		cells.emplace_back(std::vector<int>{ 60, 41, 61, 42, });
		cells.emplace_back(std::vector<int>{ 63, 44, 62, 43, });
		cells.emplace_back(std::vector<int>{ 65, 45, 64, 46, });
		cells.emplace_back(std::vector<int>{ 67, 48, 47, 66, });
		cells.emplace_back(std::vector<int>{ 69, 51, 50, 68, 49, });
		cells.emplace_back(std::vector<int>{ 53, 52, 71, 70, });
		cells.emplace_back(std::vector<int>{ 72, 73, });
		cells.emplace_back(std::vector<int>{ 54, 55, });
		cells.emplace_back(std::vector<int>{ 57, 74, 56, });
		cells.emplace_back(std::vector<int>{ 76, 58, 59, 75, });
		cells.emplace_back(std::vector<int>{ 77, });
		cells.emplace_back(std::vector<int>{ 78, 79, 95, 96, });
		cells.emplace_back(std::vector<int>{ 97, 81, 80, });
		cells.emplace_back(std::vector<int>{ 84, 83, 82, });
		cells.emplace_back(std::vector<int>{ 86, 85, 98, });
		cells.emplace_back(std::vector<int>{ 99, 87, 88, });
		cells.emplace_back(std::vector<int>{ 89, 100, 101, });
		cells.emplace_back(std::vector<int>{ 90, 102, });
		cells.emplace_back(std::vector<int>{ 103, 91, 104, });
		cells.emplace_back(std::vector<int>{ 105, 92, });
		cells.emplace_back(std::vector<int>{ 106, 94, 107, 93, });
		cells.emplace_back(std::vector<int>{ 108, });
	}
#endif
#if 0
	{
		cells.emplace_back(std::vector<int>{23, 1, 21, 22, 41, 2, 43, 0, 20, 42, });
		cells.emplace_back(std::vector<int>{24, 26, 3, 45, 44, 25, 46, 4, });
		cells.emplace_back(std::vector<int>{5, 27, 29, 48, 28, 8, 6, 47, 49, 7, });
		cells.emplace_back(std::vector<int>{53, 51, 52, 50, 32, 9, 31, 30, 10, });
		cells.emplace_back(std::vector<int>{54, 11, 13, 12, 33, 34, 73, });
		cells.emplace_back(std::vector<int>{57, 35, 36, 16, 14, 37, 56, 15, 55, });
		cells.emplace_back(std::vector<int>{39, 38, 77, 40, 19, 58, 17, 59, 18, });
		cells.emplace_back(std::vector<int>{60, 97, 61, 62, 78, 79, 80, 95, 96, });
		cells.emplace_back(std::vector<int>{84, 65, 63, 81, 64, 83, 82, });
		cells.emplace_back(std::vector<int>{67, 86, 85, 68, 87, 98, 66, });
		cells.emplace_back(std::vector<int>{69, 99, 89, 71, 100, 101, 70, 88, });
		cells.emplace_back(std::vector<int>{72, 103, 90, 102, });
		cells.emplace_back(std::vector<int>{91, 105, 74, 104, 92, });
		cells.emplace_back(std::vector<int>{76, 106, 94, 107, 93, 108, 75, });
	}
#endif

	std::vector<double> cell_black_values;
	std::vector<double> cell_green_values;
	for (const auto& cell : cells) {
		double black_val = 0.0;
		double green_val = 0.0;
		int count = 0;
		for (int led : cell) {
			black_val += black_values[led][1];
			green_val += green_values[led][1];
			++count;
		}
		black_val /= count;
		green_val /= count;
		cell_black_values.push_back(black_val);
		cell_green_values.push_back(green_val);
	}

	std::vector<std::vector<bool>> original_bits_per_cell;
	for (int i = 0; i < cells.size(); i++) {
		original_bits_per_cell.push_back(getPRBS7(100, i));
	}
	std::vector<std::vector<bool>> received_bits_per_cell;
	received_bits_per_cell.resize(cells.size());

	std::cout << "black values\n";
	for (double val : cell_black_values) {
		std::cout << val << "\n";
	}
	std::cout << "green values\n";
	for (double val : cell_green_values) {
		std::cout << val << "\n";
	}

	{
		cap.set(cv::CAP_PROP_POS_FRAMES, GREEN_FRAME);
		cv::Mat frame;
		bool ret = cap.read(frame);
		if (!ret) {
			throw std::runtime_error("Failed to read frame");
		}
		cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);

		int i = 0;
		for (const auto& box : transformed_boxes) {
			cv::Scalar color(255, 255, 255);
			cv::line(frame, box[0], box[1], color, 5);
			cv::line(frame, box[1], box[3], color, 5);
			cv::line(frame, box[3], box[2], color, 5);
			cv::line(frame, box[2], box[0], color, 5);
			++i;
		}
		cv::imshow("out", frame);
		cv::waitKey();
	}

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
			cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);

			int cell_index = 0;
			for (const auto& cell : cells) {
				double g_val = 0.0f;
				int count = 0;
				for (int led : cell) {
					auto color = getKeyColor(transformed_boxes[led], frame, H_inv, mask);
					g_val += color[1];
					++count;
				}
				g_val /= count;
				received_bits_per_cell[cell_index].push_back(isOn(g_val, cell_black_values[cell_index], cell_green_values[cell_index]));
				++cell_index;
			}

			//cv::imshow("image", frame);
			//cv::waitKey();
		}
	}

	std::cout << "original: ";
	int TOTAL_BITCOUNT = 0;
	for (const auto& bits : original_bits_per_cell) {
		for (bool b : bits) {
			std::cout << b ? '1' : '0';
			++TOTAL_BITCOUNT;
		}
	}
	std::cout << "\n";

	std::cout << "received: ";
	for (const auto& bits : received_bits_per_cell) {
		for (bool b : bits) {
			std::cout << b ? '1' : '0';

		}
	}
	std::cout << "\n";

	int errors = 0;
	for (int i = 0; i < original_bits_per_cell.size(); ++i) {
		for (int j = 0; j < original_bits_per_cell[i].size(); ++j) {
			if (original_bits_per_cell[i][j] != received_bits_per_cell[i][j]) {
				++errors;
			}
		}
	}

	std::cout << "error count: " << errors << "\n";
	std::cout << "ber: " << double(errors) * 100.0 / double(TOTAL_BITCOUNT) << "%\n";

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

	return 0;
}