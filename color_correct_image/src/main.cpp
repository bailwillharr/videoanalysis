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
#include <filesystem>

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

struct Color16 {
	uint16_t r;
	uint16_t g;
	uint16_t b;
};

static void writeColorsAsCsv(const std::filesystem::path& filename, const std::vector<std::array<Color16, 105>>& colors) {
	assert(colors.size() == 512);

	std::ofstream file(filename, std::ios::trunc | std::ios::out);
	if (!file) {
		throw std::runtime_error("Failed to open file for writing");
	}

	for (const auto& key_colors : colors) {
		for (int i = 0; i < key_colors.size(); ++i) {
			Color16 color = key_colors[i];
			file << color.r << "," << color.g << "," << color.b;
			if (i < key_colors.size() - 1) {
				file << ",";
			}
			else {
				file << "\n";
			}
		}
	}
}

static std::vector<std::array<Color16, 105>> readColorsFromCsv(const std::filesystem::path& filename) {
	std::ifstream file(filename);
	if (!file) {
		throw std::runtime_error("Failed to open file for reading");
	}

	std::vector<std::array<Color16, 105>> result;
	result.reserve(512);

	std::string line;
	while (std::getline(file, line)) {
		std::array<Color16, 105> key_colors{};

		std::stringstream ss(line);
		std::string value;
		int idx = 0;

		while (std::getline(ss, value, ',')) {
			if (idx >= 105 * 3) {
				throw std::runtime_error("Too many values in CSV line");
			}

			uint16_t v = static_cast<uint16_t>(std::stoul(value));

			int color_index = idx / 3;
			int channel = idx % 3;

			if (channel == 0) key_colors[color_index].r = v;
			else if (channel == 1) key_colors[color_index].g = v;
			else key_colors[color_index].b = v;

			++idx;
		}

		if (idx != 105 * 3) {
			throw std::runtime_error("Invalid number of values in CSV line");
		}

		result.push_back(key_colors);
	}

	if (result.size() != 512) {
		throw std::runtime_error("Invalid number of rows in CSV");
	}

	return result;
}

struct Color8 {
	uint8_t r;
	uint8_t g;
	uint8_t b;
};

// i is between 0 and 511 inclusive
static Color8 getColor(uint32_t i)
{
	Color8 c{};

	// 8 permutations per channel for a total of 512 colours

	uint8_t b3 = i % 8;
	i /= 8;
	uint8_t g3 = i % 8;
	i /= 8;
	uint8_t r3 = i % 8;

	c.r = (r3 * 255 + 3) / 7;
	c.g = (g3 * 255 + 3) / 7;
	c.b = (b3 * 255 + 3) / 7;

	return c;
}

int main()
{
	//struct Color16 {
	//	uint16_t r;
	//	uint16_t g;
	//	uint16_t b;
	//};
	// colors type is std::vector<std::array<Color16, 105>>
	auto colors = readColorsFromCsv("F:\\project\\colors.csv");

	cv::Mat image_in = cv::imread("F:\\project\\recv.png", cv::IMREAD_UNCHANGED);
	if (image_in.empty()) {
		throw std::runtime_error("Failed to read mask image");
	}
	// image_in format is CV_16UC3 // 16 bits per channel!
	assert(image_in.type() == CV_16UC3);

	constexpr int HEIGHT = 174;
	constexpr int WIDTH = 128;

	cv::Mat image_out(HEIGHT, WIDTH, CV_8UC3);

	size_t pixel_index = 0;
	for (int y = 0; y < HEIGHT; ++y) {
		for (int x = 0; x < WIDTH; ++x) {

			const size_t key_idx = pixel_index % 105;

			// access a given color with colors[color_index][key_idx]

			// Read input pixel (assumes 16-bit 3-channel image)
			const cv::Vec3w in_px = image_in.at<cv::Vec3w>(y, x);

			uint32_t best_dist = std::numeric_limits<uint32_t>::max();
			size_t best_index = 0;

			// Iterate over all palette entries
			for (size_t color_index = 0; color_index < colors.size(); ++color_index) {
				const auto& ref = colors[color_index][key_idx];

				int dr = int(in_px[2]) - int(ref.r); // OpenCV uses BGR
				int dg = int(in_px[1]) - int(ref.g);
				int db = int(in_px[0]) - int(ref.b);

				uint32_t dist = dr * dr + dg * dg + db * db;

				if (dist < best_dist) {
					best_dist = dist;
					best_index = color_index;
				}
			}

			// Map best index to 8-bit palette color
			Color8 c8 = getColor(static_cast<uint32_t>(best_index));

			cv::Vec3b out_px;
			out_px[0] = c8.b;
			out_px[1] = c8.g;
			out_px[2] = c8.r;

			image_out.at<cv::Vec3b>(y, x) = out_px;

			++pixel_index;
		}
	}

	cv::imwrite("F:\\project\\corrected.png", image_out);
	cv::imshow("corrected", image_out);
	cv::waitKey();

	return 0;
}









