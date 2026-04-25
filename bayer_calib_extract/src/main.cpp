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
	assert(colors.size() == 128 * 3);

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

static void onMouse(int event, int x, int y, int, void* data)
{
	auto pts = static_cast<std::vector<cv::Point2f>*>(data);

	if (event == cv::EVENT_LBUTTONDOWN) {
		pts->emplace_back(x, y);
		std::cout << "Clicked: " << x << ", " << y << std::endl;
	}
}

static Box transformAABB(const Box& box, const cv::Mat& H)
{
	// Get the four corners of the input box (in image coordinates)
	std::vector<cv::Point2d> corners = {
		cv::Point2d(box.x,          box.y),           // top-left
		cv::Point2d(box.x + box.w,  box.y),           // top-right
		cv::Point2d(box.x + box.w,  box.y + box.h),   // bottom-right
		cv::Point2d(box.x,          box.y + box.h)    // bottom-left
	};

	double minX = INFINITY;
	double minY = INFINITY;
	double maxX = -INFINITY;
	double maxY = -INFINITY;

	for (const auto& pt : corners) {
		// Apply homography: point' = H * [x, y, 1]^T
		cv::Mat p = (cv::Mat_<double>(3, 1) << pt.x, pt.y, 1.0);
		cv::Mat p_transformed = H * p;

		double x = p_transformed.at<double>(0, 0);
		double y = p_transformed.at<double>(1, 0);
		double w = p_transformed.at<double>(2, 0);

		if (w != 0.0) {
			x /= w;
			y /= w;
		}

		minX = std::min(minX, x);
		minY = std::min(minY, y);
		maxX = std::max(maxX, x);
		maxY = std::max(maxY, y);
	}

	// Convert to integer AABB and make it "liberal" (slightly larger)
	// to be safe due to floating-point precision and perspective effects
	int new_x = static_cast<int>(std::floor(minX)) - 1;
	int new_y = static_cast<int>(std::floor(minY)) - 1;
	int new_w = static_cast<int>(std::ceil(maxX)) - new_x + 1;
	int new_h = static_cast<int>(std::ceil(maxY)) - new_y + 1;

	// Optional: add a small safety margin
	new_x -= 2;
	new_y -= 2;
	new_w += 4;
	new_h += 4;

	return { new_x, new_y, new_w, new_h };
}

int main()
{

	/*
	const auto video_paths = std::vector{ "F:\\project\\calib1.raw", "F:\\project\\calib2.raw" ,"F:\\project\\calib3.raw" ,"F:\\project\\calib4.raw" };
	std::string bboxes_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\bboxes.csv";
	std::string mask_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\mask3.png";
	constexpr int FRAMES = 128;
	constexpr int FRAME_SKIP = 2;
	constexpr auto START_FRAMES = std::array{ 3, 3, 3, 3 };
	*/

	const auto video_paths = std::array{ "F:\\project\\red.raw", "F:\\project\\green.raw" ,"F:\\project\\blue.raw" };
	std::string bboxes_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\bboxes.csv";
	std::string mask_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\mask3.png";
	constexpr int FRAMES = 128;
	constexpr int FRAME_SKIP = 2;
	constexpr auto START_FRAMES = std::array{ 2,1,1 };

	constexpr int CAMERA_WIDTH = 1536;
	constexpr int CAMERA_HEIGHT = 864;
	constexpr size_t FRAME_SIZE = CAMERA_WIDTH * CAMERA_HEIGHT * 2;

	cv::Mat mask = cv::imread(mask_path);
	if (mask.empty()) {
		throw std::runtime_error("Failed to read mask image");
	}

	auto boxes = loadCsvBoxes(bboxes_path);

	{
		// Read a single frame
		std::vector<uint16_t> frame_buffer(CAMERA_WIDTH * CAMERA_HEIGHT);
		std::ifstream file(video_paths[0], std::ios::binary);
		file.seekg(FRAME_SIZE * (0), std::ios::beg);
		file.read(reinterpret_cast<char*>(frame_buffer.data()), FRAME_SIZE);
		if (!file) {
			std::cerr << "Failed to read frame\n";
			return 1;
		}
		cv::Mat bayer16(CAMERA_HEIGHT, CAMERA_WIDTH, CV_16UC1, frame_buffer.data());
		cv::Mat rgb(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3);
		cv::cvtColor(bayer16, rgb, cv::COLOR_BayerRG2BGR);


#if 0
		cv::Mat img1 = rgb.clone();   // camera image
		cv::Mat img2 = mask.clone();  // mask

		std::vector<cv::Point2f> pts1, pts2;

		cv::imshow("img1", img1);
		cv::setMouseCallback("img1", onMouse, &pts1);

		cv::imshow("img2", img2);
		cv::setMouseCallback("img2", onMouse, &pts2);

		cv::waitKey(0);

		cv::Mat H = cv::findHomography(pts2, pts1); // mask → rgb
#else
		cv::Mat H = (cv::Mat_<double>(3, 3) <<
			0.9801130534236812, -0.01888928686102932, 72.04139151320041,
			0.01906119163634, 0.9814887539754136, -92.98018876577001,
			-7.058406243879896e-06, -5.031016992413685e-06, 1.0
			);
#endif

		cv::Mat warped_mask;
		cv::warpPerspective(
			mask,
			warped_mask,
			H,
			rgb.size(),
			cv::INTER_LINEAR,
			cv::BORDER_CONSTANT,
			cv::Scalar(0, 0, 0)
		);

		for (auto& box : boxes) {
			box = transformAABB(box, H);
		}

		mask = warped_mask.clone();

#if 0
		for (int y = 0; y < CAMERA_HEIGHT; ++y) {
			for (int x = 0; x < CAMERA_WIDTH; ++x) {
				if (warped_mask.at<cv::Vec3b>(y, x)[1] > 128) {
					rgb.at<cv::Vec3w>(y, x) = {65535, 65535, 65535};
				}
			}
		}

		for (const auto& box : boxes) {
			cv::rectangle(rgb, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(65535, 0, 0));
		}

		cv::imshow("overlay", rgb);
		cv::waitKey(0);

		std::cout << "SAVE THIS!\n" << H << "\n";

#endif
	}

	std::vector<std::array<Color16, 105>> colors{}; // [color_idx][key_idx]

	std::vector<uint16_t> frame_buffer(CAMERA_WIDTH * CAMERA_HEIGHT);
	for (int video_idx = 0; video_idx < (int)video_paths.size(); ++video_idx) {

		std::ifstream file(video_paths[video_idx], std::ios::binary);
		const auto start_frame = START_FRAMES[video_idx];
		for (int frame_idx = start_frame; frame_idx < start_frame + (FRAMES * FRAME_SKIP); frame_idx += FRAME_SKIP) {
			// Read a single frame
			file.seekg(FRAME_SIZE * (frame_idx), std::ios::beg);
			file.read(reinterpret_cast<char*>(frame_buffer.data()), FRAME_SIZE);
			if (!file) {
				std::cerr << "Failed to read frame\n";
				break;
			}


			cv::Mat bayer16(CAMERA_HEIGHT, CAMERA_WIDTH, CV_16UC1, frame_buffer.data());

#if 0
			cv::Mat rgb(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3);
			cv::cvtColor(bayer16, rgb, cv::COLOR_BayerRG2BGR);
			std::cout << frame_idx << "\n";
			cv::imshow("rgb", rgb);
			cv::imwrite("frame.png", bayer16);
			cv::waitKey();
#endif

			colors.emplace_back();

			int box_idx = 0;
			for (const auto& box : boxes) {

				uint32_t sum_of_reds = 0;
				int red_count = 0;
				pixelsInKey(box, mask, Channel::RED, bayer16, [&](int y, int x) {
					sum_of_reds += bayer16.at<uint16_t>(y, x);
					red_count += 1;
					});
				assert(red_count != 0);
				const uint32_t average_red = (sum_of_reds / red_count);

				uint32_t sum_of_greens = 0;
				int green_count = 0;
				pixelsInKey(box, mask, Channel::GREEN, bayer16, [&](int y, int x) {
					sum_of_greens += bayer16.at<uint16_t>(y, x);
					green_count += 1;
					});
				assert(green_count != 0);
				const uint32_t average_green = (sum_of_greens / green_count);

				uint32_t sum_of_blues = 0;
				int blue_count = 0;
				pixelsInKey(box, mask, Channel::BLUE, bayer16, [&](int y, int x) {
					sum_of_blues += bayer16.at<uint16_t>(y, x);
					blue_count += 1;
					bayer16.at<uint16_t>(y, x) = 65535;
					});
				assert(blue_count != 0);
				const uint32_t average_blue = (sum_of_blues / blue_count);

				assert(average_red < 65536);
				assert(average_green < 65536);
				assert(average_blue < 65536);

				colors.back()[box_idx] = Color16{
					.r = static_cast<uint16_t>(average_red),
					.g = static_cast<uint16_t>(average_green),
					.b = static_cast<uint16_t>(average_blue)
				};

				++box_idx;
			}
		}
	}

	writeColorsAsCsv("F:\\project\\red_green_blue.csv", colors);

	return 0;
}









