#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <span>
#include <array>
#include <filesystem>

#include <opencv2/opencv.hpp>

using namespace std;

// Function to find nearest cube index
tuple<int, int, int> find_nearest_cube_index(const array<double, 3>& measured_rgb,
	const array<array<array<array<double, 3>, 4>, 8>, 4>& cube)
{
	double min_distance = 1e9; // initialize with a large number
	int best_x = 0, best_y = 0, best_z = 0;

	for (int x = 0; x < 4; ++x) {
		for (int y = 0; y < 8; ++y) {
			for (int z = 0; z < 4; ++z) {
				double dr = measured_rgb[0] - cube[x][y][z][0];
				double dg = measured_rgb[1] - cube[x][y][z][1];
				double db = measured_rgb[2] - cube[x][y][z][2];
				double distance = sqrt(dr * dr + dg * dg + db * db);

				if (distance < min_distance) {
					min_distance = distance;
					best_x = x;
					best_y = y;
					best_z = z;
				}
			}
		}
	}

	return make_tuple(best_x, best_y, best_z);
}

// cv::Vec3b is BGR
static std::vector<cv::Vec3b> loadColorData(const std::string& filename) {
	std::vector<cv::Vec3b> data{};
	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open file: " + filename);
	}

	std::string line;

	// Skip header line
	if (!std::getline(file, line)) {
		return data; // empty file
	}

	// Parse data lines
	while (std::getline(file, line)) {
		if (line.empty()) continue;

		std::stringstream ss(line);
		std::string token;

		cv::Vec3b col;

		if (std::getline(ss, token, ',')) col[2] = std::stoi(token);
		if (std::getline(ss, token, ',')) col[1] = std::stoi(token);
		if (std::getline(ss, token, ',')) col[0] = std::stoi(token);

		data.push_back(col);
	}

	return data;
}

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

static cv::Vec3b findAvgColorWiithMask(const cv::Mat& img, const cv::Mat& mask, const std::array<cv::Point2f, 4>& box, const cv::Mat& H_inv) {
	std::vector<cv::Vec3b> colors{};
	pixelsInQuad(box, img, [&](int x, int y) {
		auto coords = lookupMaskCoordinate(x, y, H_inv);
		//std::cout << "test\n";
		if (mask.at<cv::Vec3b>(coords[1], coords[0])[1] == 255) {
			colors.push_back(img.at<cv::Vec3b>(y, x));
		}
		});

	// compute average
	return averageColor(colors);
}

// Computes squared Euclidean distance between two BGR colors
static inline int colorDistanceSq(const cv::Vec3b& a, const cv::Vec3b& b)
{
	int db = int(a[0]) - int(b[0]);
	int dg = int(a[1]) - int(b[1]);
	int dr = int(a[2]) - int(b[2]);
	return db * db + dg * dg + dr * dr;
}

// Finds closest color from a list
int findClosestColor(const cv::Vec3b& inputColor, const std::array<cv::Vec3b, 128>& palette)
{
	int bestIndex = 0;
	int bestDist = INT_MAX;

	for (int i = 0; i < (int)palette.size(); i++)
	{
		int dist = colorDistanceSq(inputColor, palette[i]);
		if (dist < bestDist)
		{
			bestDist = dist;
			bestIndex = i;
		}
	}

	return bestIndex;
}


int main()
{
	std::filesystem::path images_dir = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\extracted_frames2";
	std::string mask_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\reception\\mask2.png";
	std::string bboxes_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\bboxes.csv";

	std::vector<cv::Mat> images{};
	for (int i = 0; i < 128; ++i) {
		int frame = 928 + (i * 24);
		std::string filename = std::format("frame_{:06}.png", frame);
		images.emplace_back(cv::imread((images_dir / filename).string()));
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
	dstPnts[0] = cv::Point2f(38.9, 48.3);
	dstPnts[1] = cv::Point2f(2010.3, -20.6);
	dstPnts[2] = cv::Point2f(54.3, 1114.1);
	dstPnts[3] = cv::Point2f(2022.2, 1126.5);
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

	std::vector<std::array<cv::Vec3b, 128>> calibration_data{};
	for (int i = 0; i < 109; ++i) {
		calibration_data.push_back({});
		auto& data = calibration_data.back();
		for (int j = 0; j < 128; ++j) {
			data[j] = findAvgColorWiithMask(images[j], mask, transformed_boxes[i], H_inv);
		}
	}

	std::string received_text_csv = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\reception\\text\\text_colors.csv";
	auto received_text_colors = loadColorData(received_text_csv);

	std::string out_str{};

	for (int i = 0; i < received_text_colors.size(); ++i) {

		std::cout << i << "\n";

		int key_index = i % 109;
		const auto& palette = calibration_data[key_index];

		int res = findClosestColor(received_text_colors[i], palette);
		out_str.push_back(res);
	}

	{
		std::ofstream text_output("text_output.txt");
		if (!text_output) {
			throw std::runtime_error("Failed to create file for writing");
		}
		text_output << out_str;
	}

	return 0;
}