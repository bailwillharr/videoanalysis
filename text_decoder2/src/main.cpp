#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <span>
#include <array>
#include <bit>

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

static auto getSections()
{
	std::array<std::vector<int>, 8> sections{};
	sections[0] = {
	0,
	1,
	2,
	3,
	4,
	20,
	21,
	22,
	23,
	24,
	25,
	41,
	42,
	43,
	44,
	45,
	46,
	};
	sections[1] = {
	5,
	6,
	7,
	8,
	9,
	10,
	11,
	12,
	26,
	27,
	28,
	29,
	30,
	31,
	32,
	33,
	47,
	48,
	49,
	50,
	51,
	52,
	53,
	73,
	};
	sections[2] = {
	13,
	14,
	15,
	34,
	35,
	36,
	54,
	55,
	56,
	};
	sections[3] = {
	16,
	17,
	18,
	19,
	37,
	38,
	39,
	40,
	57,
	58,
	59,
	77,
	};
	sections[4] = {
	60,
	61,
	62,
	63,
	64,
	65,
	78,
	79,
	80,
	81,
	82,
	83,
	95,
	96,
	97,
	98,
	};
	sections[5] = {
	66,
	67,
	68,
	69,
	70,
	71,
	72,
	84,
	85,
	86,
	87,
	88,
	89,
	90,
	99,
	100,
	101,
	102,
	};
	sections[6] = {
	91,
	103,
	104,
	105,
	};
	sections[7] = {
	74,
	75,
	76,
	92,
	93,
	94,
	106,
	107,
	108,
	};
	return sections;
}

auto getIndexToSection(const std::array<std::vector<int>, 8>& sections) {
	std::map<int, int> map{};
	int section_index = 0;
	for (const auto& section : sections) {
		for (const int i : section) {
			map.emplace(i, section_index);
		}
		++section_index;
	}
	std::vector<int> res{};
	for (const auto& [i, section_index] : map) {
		res.push_back(section_index);
	}
	return res;
}

static int lookupIndexFromColor(const std::vector<cv::Vec3b>& matching_colors, cv::Vec3b color) {
	int bestIndex = -1;
	int bestDist = std::numeric_limits<int>::max();

	for (int i = 0; i < matching_colors.size(); ++i) {
		const cv::Vec3b& c = matching_colors[i];

		// Compute squared Euclidean distance
		int db = int(c[0]) - int(color[0]);
		int dg = int(c[1]) - int(color[1]);
		int dr = int(c[2]) - int(color[2]);

		int dist = db * db + dg * dg + dr * dr;

		if (dist < bestDist) {
			bestDist = dist;
			bestIndex = i;
		}
	}

	if (bestIndex < 0) {
		throw std::runtime_error("Couldn't find best index");
	}
	if (bestIndex > 7) {
		throw std::runtime_error("bestIndex shouldn't be higher than 7");
	}

	return bestIndex;  // index of the closest matching color
}

int main()
{
	std::string video_path = "C:\\Users\\Bailey\\Documents\\University\\L4\\project\\Masters_Project\\bailey\\reception\\shorttext\\shorttext.mkv";
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

#if 0
	cv::Mat H = (cv::Mat_<double>(3, 3) <<
		0.9429, 0.0060, 64.2086,
		-0.0292, 0.9881, 24.4185,
		-0.0000, 0.0000, 1.0000
		);
#else

	cv::Mat H{};
	std::array<cv::Point2f, 4> srcPnts{};
	srcPnts[0] = cv::Point2f(0, 0);
	srcPnts[1] = cv::Point2f(1919, 0);
	srcPnts[2] = cv::Point2f(0, 1079);
	srcPnts[3] = cv::Point2f(1919, 1079);
	std::array<cv::Point2f, 4> dstPnts{};
	dstPnts[0] = cv::Point2f(61.8, 24.2);
	dstPnts[1] = cv::Point2f(2054, -36.4);
	dstPnts[2] = cv::Point2f(71, 1087.6);
	dstPnts[3] = cv::Point2f(2050.8, 1129.1);
	H = cv::findHomography(srcPnts, dstPnts);
	cv::Mat H_inv = H.inv();
#endif
	//H = H.inv();

	//H.at<double>(0, 1) *= -1.0;
	//H.at<double>(1, 1) *= -1.0;
	//H.at<double>(2, 1) *= -1.0;

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

	auto sections = getSections();
	auto index_to_sections = getIndexToSection(sections);


	// calibration

	std::vector<std::vector<cv::Vec3b>> measured_colors_per_section(8);
	{
		constexpr int START_FRAME = 1587 - 24;

		cv::Mat frame;
		for (int i = 0; i < 8; ++i) {
			cap.set(cv::CAP_PROP_POS_FRAMES, START_FRAME + i * 48);
			{
				bool ret = cap.read(frame);
				if (!ret) {
					throw std::runtime_error("Failed to read frame");
				}
			}

			for (int section_index = 0; section_index < 8; ++section_index) {
				// find average colour
				std::vector<cv::Vec3b> colors{};
				for (const int box_index : sections[section_index]) {
					pixelsInQuad(transformed_boxes[box_index], frame, [&](int x, int y) {
						auto coords = lookupMaskCoordinate(x, y, H_inv);
						//std::cout << "test\n";
						if (mask.at<cv::Vec3b>(coords[1], coords[0])[1] == 255) {
							colors.push_back(frame.at<cv::Vec3b>(y, x));
						}
						});
				}
				measured_colors_per_section[section_index].push_back(averageColor(colors));
			}
		}
		std::cout << "BREAK\n";
	}



	// display calibration data

	{

		cap.set(cv::CAP_PROP_POS_FRAMES, 3360);
		cv::Mat frame;
		bool ret = cap.read(frame);
		if (!ret) {
			throw std::runtime_error("Failed to read frame");
		}

		cv::Mat out(1080, 1920, CV_8UC3);
		for (int x = 0; x < 1920; ++x) {
			for (int y = 0; y < 1080; ++y) {
				auto& ref = out.at<cv::Vec3b>(y, x);
				ref[0] = 0;
				ref[1] = 0;
				ref[2] = 0;
			}
		}

		for (int j = 0; j < 8; ++j) {
			int i = 0;
			for (const auto& box : transformed_boxes) {
				int section_index = index_to_sections[i];
				cv::Scalar color = measured_colors_per_section[section_index][j];
				cv::line(out, box[0], box[1], color, 5);
				cv::line(out, box[1], box[3], color, 5);
				cv::line(out, box[3], box[2], color, 5);
				cv::line(out, box[2], box[0], color, 5);
				++i;
			}
			cv::imshow("out", out);
			cv::waitKey();
		}
	}



	// decode text
	std::vector<char> output_text{};
	std::vector<bool> parities{};
	{
		constexpr int START_FRAME = 2425;
		constexpr int FRAMES = 246;
		// each frame encodes three characters
		cv::Mat frame;
		for (int i = 0; i < FRAMES; ++i) {
			cap.set(cv::CAP_PROP_POS_FRAMES, START_FRAME + (i * 24));
			{
				bool ret = cap.read(frame);
				if (!ret) {
					throw std::runtime_error("Failed to read frame");
				}
			}

			char c1{}, c2{}, c3{};
			bool p1{}, p2{}, p3{};

			for (int section_index = 0; section_index < 8; ++section_index) {
				// find average colour
				std::vector<cv::Vec3b> colors{};
				for (const int box_index : sections[section_index]) {
					pixelsInQuad(transformed_boxes[box_index], frame, [&](int x, int y) {
						auto coords = lookupMaskCoordinate(x, y, H_inv);
						//std::cout << "test\n";
						if (mask.at<cv::Vec3b>(coords[1], coords[0])[1] == 255) {
							colors.push_back(frame.at<cv::Vec3b>(y, x));
						}
						});
				}
				int best_index = lookupIndexFromColor(measured_colors_per_section[section_index], averageColor(colors));

				if (section_index < 7) {
					if (((best_index >> 0) & 1) == 1) {
						c1 |= (1 << section_index);
					}
					if (((best_index >> 1) & 1) == 1) {
						c2 |= (1 << section_index);
					}
					if (((best_index >> 2) & 1) == 1) {
						c3 |= (1 << section_index);
					}
				}
				else {
					if (((best_index >> 0) & 1) == 1) {
						p1 = (1 << section_index);
					}
					if (((best_index >> 1) & 1) == 1) {
						p2 = (1 << section_index);
					}
					if (((best_index >> 2) & 1) == 1) {
						p3 = (1 << section_index);
					}
				}
			}
			output_text.push_back(c1);
			output_text.push_back(c2);
			output_text.push_back(c3);
			parities.push_back(p1);
			parities.push_back(p2);
			parities.push_back(p3);
		}
	}

	std::string output_text_as_string{};
	int i = 0;
	int errors = 0;
	for (const char c : output_text) {
		output_text_as_string.push_back(c);
		// check parity of char
		if ((std::popcount(static_cast<unsigned char>(c)) & 1) != parities[i]) {
			++errors;
		}
		++i;
	}

	std::cout << output_text_as_string << "\n";

	std::cout << "Errors: " << errors << "\n";

	return 0;
}