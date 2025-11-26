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
	const array<array<array<array<double, 3>, 8>, 8>, 8>& cube)
{
	double min_distance = 1e9; // initialize with a large number
	int best_x = 0, best_y = 0, best_z = 0;

	for (int x = 0; x < 8; ++x) {
		for (int y = 0; y < 8; ++y) {
			for (int z = 0; z < 8; ++z) {
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

// Trilinear interpolation
array<double, 3> interpolate_rgb(const array<double, 3>& measured_rgb,
	const array<array<array<array<double, 3>, 8>, 8>, 8>& cube,
	const array<double, 8>& channel_values)
{
	array<double, 3> result;

	// Map measured_rgb to fractional cube indices [0..7]
	array<double, 3> f_idx;
	for (int i = 0; i < 3; ++i) {
		// Find which two cube points the value lies between
		for (int j = 0; j < 7; ++j) {
			if (measured_rgb[i] <= channel_values[j + 1]) {
				double t = (measured_rgb[i] - channel_values[j]) /
					(channel_values[j + 1] - channel_values[j]);
				f_idx[i] = j + t;
				break;
			}
		}
	}

	// Floor and ceil indices
	int x0 = floor(f_idx[0]), x1 = min(x0 + 1, 7);
	int y0 = floor(f_idx[1]), y1 = min(y0 + 1, 7);
	int z0 = floor(f_idx[2]), z1 = min(z0 + 1, 7);

	// Weights
	double wx = f_idx[0] - x0;
	double wy = f_idx[1] - y0;
	double wz = f_idx[2] - z0;

	for (int c = 0; c < 3; ++c) {
		result[c] =
			cube[x0][y0][z0][c] * (1 - wx) * (1 - wy) * (1 - wz) +
			cube[x1][y0][z0][c] * wx * (1 - wy) * (1 - wz) +
			cube[x0][y1][z0][c] * (1 - wx) * wy * (1 - wz) +
			cube[x0][y0][z1][c] * (1 - wx) * (1 - wy) * wz +
			cube[x1][y1][z0][c] * wx * wy * (1 - wz) +
			cube[x1][y0][z1][c] * wx * (1 - wy) * wz +
			cube[x0][y1][z1][c] * (1 - wx) * wy * wz +
			cube[x1][y1][z1][c] * wx * wy * wz;
	}

	return result;
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

		// skip first 3 columns
		std::getline(ss, token, ',');
		std::getline(ss, token, ',');
		std::getline(ss, token, ',');
		if (std::getline(ss, token, ',')) col[2] = std::stoi(token);
		if (std::getline(ss, token, ',')) col[1] = std::stoi(token);
		if (std::getline(ss, token, ',')) col[0] = std::stoi(token);

		data.push_back(col);
	}

	return data;
}

static uchar colFromIndex(int i) {
	std::array<uchar, 8> arr{ 0, 36, 73, 109, 146, 182, 219, 255 };
	return arr.at(i);
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

static cv::Vec3b findAvgColorWiithMask(const cv::Mat& img, const cv::Mat& mask, const Box& box) {
	std::vector<cv::Vec3b> colors{};
	for (int y = box.y; y < box.y + box.h; ++y) {
		for (int x = box.x; x < box.x + box.w; ++x) {
			auto test = mask.at<cv::Vec3b>(y, x);
			if (mask.at<cv::Vec3b>(y, x)[1] == 255) {
				colors.push_back(img.at<cv::Vec3b>(y, x));
			}
		}
	}

	return averageColor(colors);
}

int main()
{
	std::filesystem::path images_dir = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\extracted_frames";
	std::string mask_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\reception\\mask2.png";
	std::string bboxes_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\bboxes.csv";

	std::vector<cv::Mat> images{};
	for (int i = 0; i < 512; ++i) {
		int frame = 1829 + (i * 24);
		std::string filename = std::format("frame_{:06}.png", frame);
		images.emplace_back(cv::imread((images_dir / filename).string()));
	}

	cv::Mat mask = cv::imread(mask_path);
	if (mask.empty()) {
		throw std::runtime_error("Failed to read mask image");
	}

	auto boxes = loadCsvBoxes(bboxes_path);

	std::vector<std::array<cv::Vec3b, 512>> calibration_data{};
	for (int i = 0; i < 109; ++i) {
		calibration_data.push_back({});
		auto& data = calibration_data.back();
		for (int j = 0; j < 512; ++j) {
			data[j] = findAvgColorWiithMask(images[j], mask, boxes[i]);
		}
	}

	std::string received_image_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\reception\\mandrill_rec.png";
	cv::Mat received_image = cv::imread(received_image_path);
	if (received_image.empty()) {
		throw std::runtime_error("Failed to read mask image");
	}

	for (int i = 0; i < 16384; ++i) {

		std::cout << i << "\n";

		int key_index = i % 109;

		std::array<double, 512> R_measured{};
		std::array<double, 512> G_measured{};
		std::array<double, 512> B_measured{};
		for (int j = 0; j < 512; ++j) {
			R_measured[j] = calibration_data[key_index][j][2];
			G_measured[j] = calibration_data[key_index][j][1];
			B_measured[j] = calibration_data[key_index][j][0];
		}

		// Build 8x8x8 cube
		array<array<array<array<double, 3>, 8>, 8>, 8> cube;
		for (int x = 0; x < 8; ++x) {
			for (int y = 0; y < 8; ++y) {
				for (int z = 0; z < 8; ++z) {
					int row = x * 64 + y * 8 + z;
					cube[x][y][z][0] = R_measured[row];
					cube[x][y][z][1] = G_measured[row];
					cube[x][y][z][2] = B_measured[row];
				}
			}
		}

		constexpr std::array<double, 8> channel_values{ 0, 36, 73, 109, 146, 182, 219, 255 };

		int x = i % 128;
		int y = i / 128;
		cv::Vec3b& px = received_image.at<cv::Vec3b>(y, x);
#if 1
		const auto [i1, i2, i3] = find_nearest_cube_index(std::array<double, 3>{static_cast<double>(px[2]), static_cast<double>(px[1]), static_cast<double>(px[0])}, cube);
		px[2] = colFromIndex(i1);
		px[1] = colFromIndex(i2);
		px[0] = colFromIndex(i3);
#else
		auto res = interpolate_rgb(std::array<double, 3>{static_cast<double>(px[2]), static_cast<double>(px[1]), static_cast<double>(px[0])}, cube, channel_values);
		px[2] = res[0];
		px[1] = res[1];
		px[0] = res[2];
#endif

	}

	cv::imshow("image", received_image);
	cv::imwrite("linear.png", received_image);
	cv::waitKey();

	return 0;
}