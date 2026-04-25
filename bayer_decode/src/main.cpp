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

static cv::Mat extractGreenFromBGGR16(const cv::Mat& bayer16) {
	CV_Assert(bayer16.type() == CV_16UC1); // ensure 16-bit single channel

	cv::Mat green = cv::Mat::zeros(bayer16.size(), bayer16.type());

	// BGGR pattern:
	// Even rows: B G B G ...
	// Odd rows:  G R G R ...

	for (int row = 0; row < bayer16.rows; ++row) {
		for (int col = 0; col < bayer16.cols; ++col) {
			if (((row % 2 == 0) && (col % 2 == 1)) || ((row % 2 == 1) && (col % 2 == 0))) {
				green.at<uint16_t>(row, col) = bayer16.at<uint16_t>(row, col);
			}
		}
	}

	return green;
}

enum class Channel {
	RED, GREEN, BLUE
};

static bool isChannelPixel(Channel channel, int y, int x) {
	bool yEven = (y % 2 == 0);
	bool xEven = (x % 2 == 0);

	if (yEven && xEven) {
		return channel == Channel::BLUE;
	}
	else if (yEven && !xEven) {
		return channel == Channel::GREEN;
	}
	else if (!yEven && xEven) {
		return channel == Channel::GREEN;
	}
	else { // !yEven && !xEven
		return channel == Channel::RED;
	}
}

static void pixelsInKey(const Box& box, const cv::Mat& mask, Channel channel, const cv::Mat& frame, std::function<void(int y, int x)> callback)
{
	for (int y = box.y; y < box.y + box.h; ++y) {
		for (int x = box.x; x < box.x + box.w; ++x) {
			if (isChannelPixel(channel, y, x)) {
				if (mask.at<cv::Vec3b>(y, x)[1] >= 127) {
					callback(y, x);
				}
			}
		}
	}
}

int main()
{
	std::string video_path = "F:\\project\\image.raw";
	std::string bboxes_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\bboxes.csv";
	std::string mask_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\mask3.png";

	constexpr int FRAMES = 213;
	constexpr int FRAME_SKIP = 2;
	constexpr int START_FRAME = 5;

	constexpr int CAMERA_WIDTH = 1536;
	constexpr int CAMERA_HEIGHT = 864;
	constexpr size_t FRAME_SIZE = CAMERA_WIDTH * CAMERA_HEIGHT * 2;

	constexpr int IMAGE_WIDTH = 128;
	constexpr int IMAGE_HEIGHT = 174;

	cv::Mat mask = cv::imread(mask_path);
	if (mask.empty()) {
		throw std::runtime_error("Failed to read mask image");
	}

	auto boxes = loadCsvBoxes(bboxes_path);

	std::ifstream file(video_path, std::ios::binary);

	cv::Mat out(IMAGE_HEIGHT, IMAGE_WIDTH, CV_16UC3);
	uint32_t current_out_pixel = 0;

	for (int frame = START_FRAME; frame < START_FRAME + (FRAMES * FRAME_SKIP); frame += FRAME_SKIP) {

		// Read a single frame
		std::vector<uint16_t> buffer(CAMERA_WIDTH * CAMERA_HEIGHT);
		file.seekg(FRAME_SIZE * (frame), std::ios::beg);
		file.read(reinterpret_cast<char*>(buffer.data()), FRAME_SIZE);
		if (!file) {
			std::cerr << "Failed to read frame\n";
			break;
		}

		for (uint16_t& value : buffer) {
			//value >>= 6; // gotta bit shift bayer
		}

		// Wrap buffer in a cv::Mat (16-bit single channel)
		cv::Mat bayer16(CAMERA_HEIGHT, CAMERA_WIDTH, CV_16UC1, buffer.data());

		//cv::Mat rgb(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3);
		//cv::cvtColor(bayer16, rgb, cv::COLOR_BayerRG2BGR);
		//std::cout << frame << "\n";
		//cv::imshow("rgb", rgb);
		//cv::waitKey();

		for (const auto& box : boxes) {
			
			uint32_t sum_of_reds = 0;
			int red_count = 0;
			pixelsInKey(box, mask, Channel::RED, bayer16, [&](int y, int x) {
				sum_of_reds += bayer16.at<uint16_t>(y, x);
				red_count += 1;
				});
			assert(red_count != 0);
			const uint32_t average_red_16bit = (sum_of_reds / red_count);
			assert(average_red_16bit < 65536);

			uint32_t sum_of_greens = 0;
			int green_count = 0;
			pixelsInKey(box, mask, Channel::GREEN, bayer16, [&](int y, int x) {
				sum_of_greens += bayer16.at<uint16_t>(y, x);
				green_count += 1;
				});
			assert(green_count != 0);
			const uint32_t average_green_16bit = (sum_of_greens / green_count);
			assert(average_green_16bit < 65536);

			uint32_t sum_of_blues = 0;
			int blue_count = 0;
			pixelsInKey(box, mask, Channel::BLUE, bayer16, [&](int y, int x) {
				sum_of_blues += bayer16.at<uint16_t>(y, x);
				blue_count += 1;
				});
			assert(blue_count != 0);
			const uint32_t average_blue_16bit = (sum_of_blues / blue_count);
			assert(average_blue_16bit < 65536);


			const int out_y = current_out_pixel / IMAGE_WIDTH;
			if (out_y >= IMAGE_HEIGHT) {
				break;
			}
			const int out_x = current_out_pixel % IMAGE_WIDTH;
			assert(out_x < IMAGE_WIDTH);
			out.at<cv::Vec3w>(out_y, out_x) = { (uint16_t)average_blue_16bit, (uint16_t)average_green_16bit, (uint16_t)average_red_16bit};
			current_out_pixel += 1;
		}

		std::cout << frame << "\n";

		//cv::imshow("frame bayer", bayer16);
		//cv::waitKey();

	}

	cv::imwrite("output.png", out);
	cv::imshow("output", out);
	cv::waitKey();

	return 0;
}









