//
// Created by xin on 2023/8/24.
//

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "TofDepthData.h"
#include <eigen3/Eigen/Dense>
#include "Resolution.h"
#include "CameraMoudleParam.h"
#include <yaml-cpp/yaml.h>
#include "config.h"
#include "file.h"
#include "utils.h"
#include "sstream"

#define EXIST(file) (access((file).c_str(), 0) == 0)
#define ERROR_PRINT(x) std::cout << "" << (x) << "" << std::endl
const float RADIAN_2_ANGLE = 180 / M_PI;
const psl::Resolution RESOLUTION = psl::Resolution::RES_640X400;
const int MAX_DEPTH = 700;
const int MAX_CHANNEL_VALUE = 255;
const int MAX_DOUBLE_CHANNEL_VALUE = 511;
const int WIDTH = 640;
const int HEIGHT = 400;


extern int access(const char *__name, int __type) __THROW __nonnull ((1));

void SunnyTof2Points(const psl::TofDepthData &tofDepthData, std::vector<Eigen::Vector3d> &points) {
    for (auto const &tof: tofDepthData.data) {
        points.push_back(Eigen::Vector3d(-tof.X, -tof.Y, tof.Z));
    }
}

void TofPointRotationAngle(std::vector<Eigen::Vector3d> &points, const float angle) {
// why this cost time more
//    Eigen::Matrix3d R;
//    R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX());
//
//    for (auto &point : points)
//    {
//        point = R * point;
//    }

    for (auto &point: points) {
        float Y = point[1];
        float Z = point[2];
        point[1] = Y * cos(angle) - Z * sin(angle);
        point[2] = Z * cos(angle) + Y * sin(angle);
    }
}

void ExchangeXYZ(std::vector<Eigen::Vector3d> &points) {
    for (auto &point: points) {
        float tempX = point[0];
        float tempY = point[1];
        float tempZ = point[2];
        point[0] = tempZ;
        point[1] = tempX;
        point[2] = tempY;
    }
}


bool GetTofPoints(const psl::TofDepthData &tof, std::vector<Eigen::Vector3d> &points, const float angle) {
    SunnyTof2Points(tof, points); // --> x,y,z
    TofPointRotationAngle(points, angle);
    ExchangeXYZ(points);
}

void ReadArray(const YAML::Node &config, std::vector<float> &array) {
    try {
        array = config.as<std::vector<float>>();
    }
    catch (...) {
        for (YAML::const_iterator it = config.begin(); it != config.end(); ++it) {
            array.push_back((*it).as<float>());
        }
    }
}

bool GetTof(std::string yamlFile, psl::TofDepthData &tof) {
    try {
        YAML::Node config;

        if (not access(yamlFile.c_str(), 0) == 0) {
            std::cout << "file not exist <" + yamlFile + ">" << std::endl;
        }
        config = YAML::LoadFile(yamlFile);

        tof.tofID = config["tofID"].as<int>();
        tof.time_stamp = config["time_stamp"].as<unsigned long>();
        tof.frameIndex = config["frameIndex"].as<unsigned long>();

        tof.height = config["height"].as<int>();
        tof.width = config["width"].as<int>();

        std::vector<float> sn(8);
        ReadArray(config["sn"], sn);

        for (int i = 0; i < 8; i++) {
            tof.sn[i] = sn[i];
        }

        static std::vector<float> data(tof.height * tof.width * 6);

        ReadArray(config["data"], data);

        int p = 0;
        for (int i = 0; i < tof.height * tof.width; i++) {
            tof.data[i].X = data.at(p++);
            tof.data[i].Y = data.at(p++);
            tof.data[i].Z = data.at(p++);
            tof.data[i].noise = data.at(p++);
            tof.data[i].grayValue = data.at(p++);
            tof.data[i].depthConfidence = data.at(p++);
        }

        return true;
    }
    catch (...) {
        std::cout << "GetTof data wrong!!!!" << std::endl;
        return false;
    }
}

bool GetCameraConfig(std::string file, psl::CameraMoudleParam &param) {
    cv::FileStorage fileStream = cv::FileStorage(file, cv::FileStorage::READ);

    if (not fileStream.isOpened()) {
        ERROR_PRINT("file not exist <" + file + ">");
        return false;
    }

    // TODO : the exception for lack option
    cv::Mat_<double> kl, dl, pl, rl;
    fileStream["Kl"] >> kl;
    fileStream["Dl"] >> dl;
    fileStream["Pl"] >> pl;
    fileStream["Rl"] >> rl;

    memcpy(param._left_camera[RESOLUTION]._K, kl.data, sizeof(param._left_camera[RESOLUTION]._K));
    memcpy(param._left_camera[RESOLUTION]._R, rl.data, sizeof(param._left_camera[RESOLUTION]._R));
    memcpy(param._left_camera[RESOLUTION]._P, pl.data, sizeof(param._left_camera[RESOLUTION]._P));
    memcpy(param._left_camera[RESOLUTION]._D, dl.data, sizeof(param._left_camera[RESOLUTION]._D));

    // TODO : the exception for lack option
    cv::Mat_<double> kr, dr, pr, rr;
    fileStream["Kr"] >> kr;
    fileStream["Dr"] >> dr;
    fileStream["Pr"] >> pr;
    fileStream["Rr"] >> rr;
    memcpy(param._right_camera[RESOLUTION]._K, kr.data, sizeof(param._right_camera[RESOLUTION]._K));
    memcpy(param._right_camera[RESOLUTION]._R, rr.data, sizeof(param._right_camera[RESOLUTION]._R));
    memcpy(param._right_camera[RESOLUTION]._P, pr.data, sizeof(param._right_camera[RESOLUTION]._P));
    memcpy(param._right_camera[RESOLUTION]._D, dr.data, sizeof(param._right_camera[RESOLUTION]._D));

    fileStream.release();

    return true;
}

bool GetParkerConfig(const std::string &configFile, ConfigParam &configParam) {
    YAML::Node configYaml = YAML::LoadFile(configFile);
    try {
        configParam.structure.leftcamera2lidar = configYaml["deeplearning"]["structure"]["leftcamera2lidar"].as<std::vector<float >>();
        configParam.structure.rightcamera2lidar = configYaml["deeplearning"]["structure"]["rightcamera2lidar"].as<std::vector<float >>();
        configParam.structure.cameraHeight = configYaml["deeplearning"]["structure"]["cameraHeight"].as<float>();
    }
    catch (const std::exception &e) {
        return ("config option <\nstructure/\n\tpose2Machine\n\tleftcamera2lidar\n\trightcamera2lidar"
                "\n\tlidarDirection\n\tlidarAngle\n\tcameraHeight\n\t> missed");
    }
    try {
        configParam.structure.cameraAngle = configYaml["deeplearning"]["structure"]["cameraAngle"].as<float>();
    }
    catch (const std::exception &e) {
        configParam.structure.cameraAngle = 10;
    }
    configParam.structure.cameraAngle = configParam.structure.cameraAngle / RADIAN_2_ANGLE;

    try {
        configParam.structure.tofAngle = configYaml["deeplearning"]["structure"]["tofAngle"].as<float>();
    }
    catch (const std::exception &e) {
        configParam.structure.tofAngle = 28;
    }
    configParam.structure.tofAngle = configParam.structure.tofAngle / RADIAN_2_ANGLE;

    try {
        configParam.structure.tofHeight = configYaml["deeplearning"]["structure"]["tofHeight"].as<float>();
    }
    catch (const std::exception &e) {
        configParam.structure.tofHeight = 0.1588;
    }
    try {
        configParam.structure.leftcamera2tof =
                configYaml["deeplearning"]["structure"]["leftcamera2tof"].as<std::vector<float >>();
        configParam.structure.rightcamera2lidar =
                configYaml["deeplearning"]["structure"]["rightcamera2tof"].as<std::vector<float >>();
        configParam.structure.poseTof2Machine =
                configYaml["deeplearning"]["structure"]["poseTof2Machine"].as<std::vector<float >>();
    }
    catch (const std::exception &e) {
        configParam.structure.leftcamera2tof = {0, 0, 0};
        configParam.structure.rightcamera2tof = {0, 0, 0};
        configParam.structure.poseTof2Machine = {0, 0, 0};
    }
}

int getBox(cv::Mat imageLeft,std::string fileName,BoxInfo& box,std::string inputSavePath) {
    // 读取图像
    cv::Mat image = imageLeft;
    int image_width = image.cols;
    int image_height = image.rows;
    std::size_t pos = inputSavePath.find_last_of(".");  // 查找最后一个点的位置
    std::string result = inputSavePath.substr(0, pos);  // 获取从开头到最后一个点之前的子字符串
    std::string detectPath = result +"/" + "detect_" + fileName;
    size_t dotPos = fileName.find_last_of(".");
    fileName.replace(dotPos, fileName.length() - dotPos, ".txt");
    std::string filePath = "/media/xin/data1/data/parker_data/result/label/label_txt/" + fileName;
    // 读取YOLOv5格式的txt文件
    std::ifstream infile(filePath);
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int class_id;
        float x_center, y_center, width, height;
        // 使用输入流 iss 读取每行文本，并将读取到的值依次赋给 class_id、x_center、y_center、width 和 height 这些变量
        if (!(iss >> class_id >> x_center >> y_center >> width >> height)) {
            break;
        }

        // 计算物体边界框的左上角和右下角坐标
        box.cx = (x_center - width / 2) * image_width;
        box.cy = (y_center - height / 2) * image_height;
        box.w = width * image_width;
        box.h = height * image_height;

        // 绘制边界框
        cv::rectangle(image, cv::Rect(box.cx, box.cy, box.w, box.h), cv::Scalar(0, 255, 0), 2);
        // 显示图像
        file_op::File::MkdirFromFile(detectPath);
        cv::imwrite(detectPath,image);
    }
    return 0;
}


void TofPointsInImage(const std::vector<Eigen::Vector3d> &points, std::vector<cv::Point> &imagePoints,
                      std::vector<Eigen::Vector3d> &pointsSelect, std::vector<float> leftcamera2tof,
                      Eigen::Matrix<double, 3, 4> P,cv::Mat imageLeft,std::string fileName,BoxInfo box,std::string inputSavePath,bool pointsSelected = false) {
    if (pointsSelected){
        getBox(imageLeft,fileName,box,inputSavePath);
    }
    imagePoints.clear();
    pointsSelect.clear();
    long i = 0;
    for (const auto &point: points) {
        ++i;
        if (point[0] < 0.01)
            continue;
        cv::Point pointTemp;
        float x = -(point[1] - leftcamera2tof[1]);
        float y = -(point[2] - leftcamera2tof[2]);
//        if (y >0.15) continue;
        float z = point[0] - leftcamera2tof[0];
        float u = P(0, 0) * x + P(0, 2) * z;
        float v = P(1, 1) * y + P(1, 2) * z;
        float scale = z * P(2, 2);
        pointTemp.x = int(u / scale);
        pointTemp.y = int(v / scale);
        // 判断该点是否在检测框内
        if (pointsSelected){
            if (pointTemp.x > box.cx && pointTemp.x < (box.cx + box.w)&& pointTemp.y > box.cy && pointTemp.y < (box.cy + box.h))
            {
                pointsSelect.push_back(point);
                imagePoints.push_back(pointTemp);
            }
        } else{
            imagePoints.push_back(pointTemp);
            pointsSelect.push_back(point);
        }

    }
}




void TofPointsInImage2DepthValues(std::vector<Eigen::Vector3d> &pointsSelect, std::vector<int> &depthValues)
{
    depthValues.clear();
//    points.clear();
//    for (int i = 0; i < pointsSelect.size(); ++i)
//    {
//        cv::Point imagePoint = imagePoints[i];
//        points.push_back(pointsSelect[i]);
//        pointsSelectedDraw.push_back(pointsSelect[i]);
//        imagePointsDraw.push_back(imagePoints[i]);
//
//    }
    for(auto const point : pointsSelect)
    {
        depthValues.push_back(int(point[0] * 100));
    }
}

void SetImageValueByPoint(cv::Mat &image, const cv::Point &point, const cv::Scalar &scalar)
{
//    int a = image.channels();
    if (image.channels() ==1)
    {

    }
    else if (image.channels() == 3)
    {
        if (point.y >= 0 && point.y < image.rows && point.x < image.cols && point.x >= 0)
        {
            image.at<cv::Vec3b>(point.y, point.x)[0] = scalar(0);
            image.at<cv::Vec3b>(point.y, point.x)[1] = scalar(1);
            image.at<cv::Vec3b>(point.y, point.x)[2] = scalar(2);
        }
    }
}

void DepthValue2Scalar(const int depthValue, cv::Scalar &scalar)
{
    int value = MIN(depthValue, MAX_DEPTH);

    if (value > MAX_DOUBLE_CHANNEL_VALUE)
    {
        scalar = cv::Scalar(0, 0, value % MAX_CHANNEL_VALUE);
    }
    else if (value > MAX_CHANNEL_VALUE)
    {
        scalar = cv::Scalar(0, value % MAX_CHANNEL_VALUE, 0);
    }
    else
    {
        scalar = cv::Scalar(value , 0, 0);
    }
}

void DrawPoints(cv::Mat &image, const std::vector<cv::Point> &imagePoints
        , const std::vector<int> &depthValues, bool selected = false)
{
    int number = 0;
    for (int i = 0; i < imagePoints.size(); ++i)
    {
        cv::Scalar scalar;
        DepthValue2Scalar(depthValues[i], scalar);
        if (selected)
        {
            cv::rectangle(image, cv::Point(imagePoints[i].x - 1, imagePoints[i].y - 1)
                    , cv::Point(imagePoints[i].x + 1, imagePoints[i].y + 1), scalar, -1);
        }
        else
        {
//            cv::circle(image, imagePoints[i], 1, scalar, -1);
            SetImageValueByPoint(image, imagePoints[i], scalar);
        }
        if (imagePoints[i].y >=0 && imagePoints[i].y < image.rows && imagePoints[i].x >=0 && imagePoints[i].x < image.cols)
        {
            number++;
        }
    }
}
std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> parts;
    std::istringstream ss(s);
    std::string part;
    while (std::getline(ss, part, delimiter)) {
        parts.push_back(part);
    }
    return parts;
}

std::string getLastTwoPathParts(const std::string &path) {
    std::vector<std::string> pathParts = split(path, '/');
    std::string result;

    if (pathParts.size() >= 2) {
        result = pathParts[pathParts.size() - 2] + "/" + pathParts[pathParts.size() - 1] + "/";
    }

    return result;
}
bool showTof(cv::Mat imageDisplay,std::vector<cv::Point> imagePoints, std::vector<int> depthValues,std::string fileName,std::string inputDir) {

    DrawPoints(imageDisplay, imagePoints, depthValues);
    {
        std::string lastTwoParts = getLastTwoPathParts(inputDir);
        std::string tofLabelPath = "/media/xin/data1/data/parker_data/tof_label/" + lastTwoParts;
        std::string file = tofLabelPath + "result_image_with_tof/" + fileName;
        file_op::File::MkdirFromFile(file);
        cv::imwrite(file, imageDisplay);
//        file = "result_image_with_tof_copy/" + imagePath;
//        file_op::File::MkdirFromFile(file);
//        cv::imwrite(file, imageDisplay);

        cv::Mat imageOut(imageDisplay.rows, imageDisplay.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        DrawPoints(imageOut, imagePoints, depthValues);

        file = tofLabelPath + "result_tof/" + fileName;
        file_op::File::MkdirFromFile(file);
        cv::imwrite(file, imageOut);
    }

//    imagePoints.clear();
//    depthValues.clear();
//    instanceManager.GetSelectPoints(imagePoints, depthValues);
//    DrawPoints(imageDisplay, imagePoints, depthValues, true);
}


bool ReadFile(std::string srcFile, std::vector<std::string> &image_files)
{
    if (not access(srcFile.c_str(), 0) == 0)
    {
        ERROR_PRINT("no such File (" + srcFile + ")");
        return false;
    }

    std::ifstream fin(srcFile.c_str());

    if (!fin.is_open())
    {
        ERROR_PRINT("read file error (" + srcFile + ")");
        return false;
    }

    std::string s;
    while (getline(fin, s))
    {
        image_files.push_back(s);
    }

    fin.close();

    return true;
}

bool ReadSyncFile(std::string srcFile, std::vector<SyncDataFile> &files, const bool &flag)
{
    if (not access(srcFile.c_str(), 0) == 0)
    {
        ERROR_PRINT("no such File (" + srcFile + ")");
        return false;
    }

    std::ifstream fin(srcFile.c_str());

    if (!fin.is_open())
    {
        ERROR_PRINT("read file error (" + srcFile + ")");
        return false;
    }

    std::string s;
    SyncDataFile syncFile;

    do
    {
        fin >> syncFile.imageLeft >> syncFile.imagePose >> syncFile.lidar
            >> syncFile.lidarPose;
        if (flag)
        {
            fin >> syncFile.tof;
            fin >> syncFile.tofPose;
        }
        files.push_back(syncFile);
    } while (fin.get() != EOF);

    if (files.size() > 1) files.pop_back();

    fin.close();

    return true;
}

bool GetData(const std::string inputDir, std::vector<SyncDataFile>& dataset, const bool &flag)
{
    std::string imagesTxt = inputDir + "/image.txt";
    std::string lidarTxt = inputDir + "/lidar.txt";
//    std::string syncTxt = inputDir + "/sync.txt";
//    std::string syncTxt = inputDir + "/test.txt";
    std::string syncTxt = inputDir + "/with_desk.txt";
    const bool synced = not EXIST(imagesTxt);
    bool binocular = false;
    std::vector<std::string> imageNameList, lidarNameList;
    std::vector<SyncDataFile> fileList;

    if (synced)
    {
        if (not ReadSyncFile(syncTxt, fileList, flag)) exit(0);
    }
    else
    {
        if (not ReadFile(imagesTxt, imageNameList)) exit(0);
        if (not ReadFile(lidarTxt, lidarNameList)) exit(0);
    }

    const size_t size =
            imageNameList.size() > 0 ? imageNameList.size() : fileList.size();

    for (size_t i = 0; i < size; ++i)
    {
        SyncDataFile item;

        if (synced)
        {
            std::string imageLeftName = fileList.at(i).imageLeft;
            fileList.at(i).imageCam0 = imageLeftName.replace(imageLeftName.find_last_of('.'), imageLeftName.length(), ".png");
            item = fileList.at(i).SetPrefix(inputDir + "/");
            if (not EXIST(item.imageLeft))
            {
                binocular = true;
                item.AddCam01Path();
            }
        }
        else
        {
            item.imageCam0 = imageNameList.at(i);
            item.imageLeft = inputDir + "/" + imageNameList.at(i);
            item.lidar = inputDir + "/" + lidarNameList[i];
            item.lidarPose = item.lidar;
            item.lidarPose = item.lidarPose.replace(item.lidar.find("lidar"), 5, "slam");
            item.imagePose = item.lidarPose;
        }

        dataset.push_back(item);
    }

    return binocular;
}

std::string GetFileNameFromPath(const std::string& path) {
    size_t lastSlashPos = path.find_last_of("/\\"); // 查找最后一个路径分隔符的位置

    if (lastSlashPos != std::string::npos) {
        std::string FileName = path.substr(lastSlashPos + 1);
        // 将 ".jpg" 替换为 ".png"
        size_t extensionPos = FileName.find(".jpg");
        return FileName.replace(extensionPos, 4, ".png");; // 提取文件名部分,并转为png
    } else {
        return path; // 如果没有路径分隔符，则返回整个路径作为文件名
    }
}

bool ReadPara(const psl::CameraParam &cameraParam
        , cv::Mat& remapX, cv::Mat& remapY)
{
    cv::Mat K, P, R, D;
    K = cv::Mat(3, 3, CV_64FC1, (unsigned char *) cameraParam._K);
    D = cv::Mat(4, 1, CV_64FC1, (unsigned char *) cameraParam._D);
//    R = cv::Mat::eye(3, 3, CV_64FC1);
    R = cv::Mat(3, 3, CV_64FC1, (unsigned char *) cameraParam._R);
    P = cv::Mat(3, 4, CV_64FC1, (unsigned char *) cameraParam._P);

    remapX.create(cv::Size(WIDTH, HEIGHT), CV_32FC1);
    remapY.create(cv::Size(WIDTH, HEIGHT), CV_32FC1);
    cv::fisheye::initUndistortRectifyMap(K, D, R, P.rowRange(0, 3).colRange(0, 3)
            , cv::Size(WIDTH, HEIGHT), CV_32F
            , remapX, remapY);
}

bool Remap(const CameraType type, cv::Mat &image,cv::Mat remapX,cv::Mat remapY)
{

    cv::setNumThreads(0);
    cv::Mat remap = image;
    cv::Mat rgb;

    if (LEFT == type)
    {
        cv::remap(image, remap, remapX, remapY
                , cv::INTER_LINEAR);
    }
    else
    {
        cv::remap(image, remap, remapX, remapY
                , cv::INTER_LINEAR);
    }
    if (1 == remap.channels())
    {
        std::vector<cv::Mat> grayGroup(3, remap);
        cv::merge(grayGroup, rgb);
    }
    else if (3 == remap.channels())
    {
        rgb = remap;
    }
    else
    {
        return false;
    }

    image = rgb;

    return true;
}

void Depth2PointCloud(const cv::Mat &depth,std::vector<Eigen::Vector3d> &cloud, bool usedTof)
{
    for (int m = 0; m < depth.rows; m++)
        for (int n = 0; n < depth.cols; n++)
        {
            // 获取深度图中(m,n)处的值
            //            ushort d = depth.ptr<ushort>(m)[n];
            //            unsigned int d = depth.at<uchar>(m,n*3);

            //使用视差图
            unsigned int d=depth.at<uchar>(m,n);

            //使用tof点
            if (usedTof)
            {
                if ( depth.at<uchar>(m,n*3+2) > 0)
                {
                    d = 511 + depth.at<uchar>(m,n*3+2);
                }
                else if (depth.at<uchar>(m,n*3 + 1) > 0)
                {
                    d = 255 + depth.at<uchar>(m,n*3+1);
                }
                else
                {
                    d = depth.at<uchar>(m,n*3);
                }
            }
                //使用深度图
            else if (depth.channels() == 3)
            {
                d = depth.at<uchar>(m,n*3) + depth.at<uchar>(m,n*3 + 1) + depth.at<uchar>(m,n*3+2);
            }

            //使用视差图
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            Point3D p;

            // 计算这个点的空间坐标
            p.z = double(d) / camera_factor;
            p.x = (n - camera_cx) * p.z / camera_fx;
            p.y = (m - camera_cy) * p.z / camera_fy;


            // 把p加入到点云中
            cloud.push_back(Eigen::Vector3d(p.x, p.y, p.z));
//            cloud->points.push_back(p);
        }
}

void getPointCloud(cv::Mat imageDisplay,std::vector<cv::Point> imagePoints, std::vector<int> depthValues,std::vector<Eigen::Vector3d> &cloud, bool usedTof){

    {
        cv::Mat imageOut(imageDisplay.rows, imageDisplay.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        DrawPoints(imageOut, imagePoints, depthValues);
        Depth2PointCloud(imageOut,cloud, usedTof);
    }
}

// 判断给定数值属于哪个区间
int findInterval(const std::vector<Interval>& intervals, double value) {
    for (int i = 0; i < intervals.size(); ++i) {
        if (value >= intervals[i].lower_bound && value <= intervals[i].upper_bound) {
            return i;  // 返回区间的索引
        }
    }
    return -1;  // 如果没有匹配的区间，返回-1
}

void drawTopView(std::string inputSavePath,std::vector<Eigen::Vector3d> cloudPoints,std::string fileName) {
    // 计算图像大小
    double minX = std::numeric_limits<double>::max();
    double maxX = -std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxY = -std::numeric_limits<double>::max();
    double minZ = std::numeric_limits<double>::max();
    double maxZ = -std::numeric_limits<double>::max();

    for (const Eigen::Vector3d &point: cloudPoints) {
        if (point.x() < minX) minX = point.x();
        if (point.x() > maxX) maxX = point.x();
        if (point.y() < minY) minY = point.y();
        if (point.y() > maxY) maxY = point.y();
        if (point.z() < minZ) minZ = point.z();
        if (point.z() > maxZ) maxZ = point.z();
    }
    double absMinX = std::abs(minX);
    double absMinY = std::abs(minY);
    double absMinZ = std::abs(minZ);

    // 遍历点云，修改每个点的值
    for (Eigen::Vector3d &point: cloudPoints) {
        point.x() += absMinX;
        point.y() += absMinY;
//        point.z() += absMinZ;
    }
    // 创建图像
    double scaleFactorW = WIDTH / (maxY - minY);
    double scaleFactorH = HEIGHT / (maxX - minX);


    cv::Mat topView(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));// 创建黑色背景图像

    // 定义区间数和区间高度差
    int numIntervals = 5; // 将高度范围分成 5 个区间
    double intervalHeight = (maxZ - minZ) / numIntervals;

    std::vector<cv::Mat> topViews;  // 存储 cv::Mat 对象的向量

    for (int i = 0; i < numIntervals; ++i) {
        cv::Mat topView(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
        topViews.push_back(topView);  // 将创建的 cv::Mat 对象添加到向量中
    }


    // 构建区间数据
    std::vector<Interval> intervals;
    for (int i = 0; i < 5; ++i) {
        Interval interval;
        interval.lower_bound = minZ + i * intervalHeight;
        interval.upper_bound = minZ + (i + 1) * intervalHeight;
        intervals.push_back(interval);
    }

    // 输出区间范围
    std::cout << "区间范围：" << std::endl;
    for (const auto& interval : intervals) {
        std::cout << "[" << interval.lower_bound << ", " << interval.upper_bound << "]" << std::endl;
    }

    // 将所有点云坐标按照高度投影到图像
    for (const Eigen::Vector3d &point: cloudPoints) {
        int interval_index = findInterval(intervals, point.z());
//        std::cout << interval_index << std::endl;
        int projectedX = static_cast<int>((point.y() - (minY + absMinY)) * scaleFactorW); // 映射到图像X坐标
        int projectedY = static_cast<int>((point.x() - (minX + absMinX)) * scaleFactorH); // 映射到图像Y坐标

        cv::Vec3b color(255, 255, 255); // 白色
        if (projectedX >= 0 && projectedX < WIDTH && projectedY >= 0 && projectedY < HEIGHT) {
            topViews[interval_index].at<cv::Vec3b>(projectedY, projectedX) = color; // 根据不同的高度区间在图像上设置颜色
            topView.at<cv::Vec3b>(projectedY, projectedX) = color; // 根据在图像上设置颜色
        }
    }

    // 保存图像
    std::size_t pos = inputSavePath.find_last_of(".");  // 查找最后一个点的位置
    std::string result = inputSavePath.substr(0, pos);  // 获取从开头到最后一个点之前的子字符串
    for (int i = 0; i < 5; ++i) {
        std::string file = result +"/" + std::to_string(i) + fileName;
        file_op::File::MkdirFromFile(file);
        cv::imwrite(file, topViews[i]);
    }

      // 显示图像
    std::string topViewPath = result +"/" + "total_" + fileName;
    file_op::File::MkdirFromFile(topViewPath);
    cv::imwrite(topViewPath, topView);
//    cv::imshow("res",topView);
//    cv::waitKey(2000);
}


void drawCloudTopView(std::string inputSavePath,std::vector<Eigen::Vector3d> cloudPoints,std::string fileName) {
    // 计算图像大小
    double minX = std::numeric_limits<double>::max();
    double maxX = -std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxY = -std::numeric_limits<double>::max();
    double minZ = std::numeric_limits<double>::max();
    double maxZ = -std::numeric_limits<double>::max();

    for (const Eigen::Vector3d &point: cloudPoints) {
        if (point.x() < minX) minX = point.x();
        if (point.x() > maxX) maxX = point.x();
        if (point.y() < minY) minY = point.y();
        if (point.y() > maxY) maxY = point.y();
        if (point.z() < minZ) minZ = point.z();
        if (point.z() > maxZ) maxZ = point.z();
    }
    double absMinX = std::abs(minX);
    double absMinY = std::abs(minY);
    double absMinZ = std::abs(minZ);

    // 遍历点云，修改每个点的值
    for (Eigen::Vector3d &point: cloudPoints) {
        point.x() += absMinX;
        point.y() += absMinY;
        point.z() += absMinZ;
    }
    // 创建图像
    double scaleFactorW = WIDTH / (maxY - minY);
    double scaleFactorH = HEIGHT / (maxX - minX);


    cv::Mat topView(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));// 创建黑色背景图像

    // 定义区间数和区间高度差
    int numIntervals = 5; // 将高度范围分成 5 个区间
    double intervalHeight = (maxZ - minZ) / numIntervals;

    std::vector<cv::Mat> topViews;  // 存储 cv::Mat 对象的向量

    for (int i = 0; i < numIntervals; ++i) {
        cv::Mat topView(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
        topViews.push_back(topView);  // 将创建的 cv::Mat 对象添加到向量中
    }


    // 构建区间数据
    std::vector<Interval> intervals;
    for (int i = 0; i < 5; ++i) {
        Interval interval;
        interval.lower_bound = minZ + i * intervalHeight;
        interval.upper_bound = minZ + (i + 1) * intervalHeight;
        intervals.push_back(interval);
    }

    // 输出区间范围
    std::cout << "区间范围：" << std::endl;
    for (const auto& interval : intervals) {
        std::cout << "[" << interval.lower_bound << ", " << interval.upper_bound << "]" << std::endl;
    }

    // 将所有点云坐标按照高度投影到图像
    for (const Eigen::Vector3d &point: cloudPoints) {
        int interval_index = findInterval(intervals, point.z());
//        std::cout << interval_index << std::endl;
        int projectedX = static_cast<int>((point.x() - (minX + absMinX)) * scaleFactorW); // 映射到图像X坐标
        int projectedY = static_cast<int>((point.y() - (minY + absMinY)) * scaleFactorH); // 映射到图像Y坐标
        int projectedZ = static_cast<int>((point.z() - (minZ + absMinZ)) * scaleFactorH); // 映射到图像Y坐标

        cv::Vec3b color(255, 255, 255); // 白色
        if (projectedX >= 0 && projectedX < WIDTH && projectedY >= 0 && projectedY < HEIGHT) {
//            topViews[interval_index].at<cv::Vec3b>(projectedY, projectedX) = color; // 根据不同的高度区间在图像上设置颜色
            topView.at<cv::Vec3b>(projectedY, projectedZ) = color; // 根据在图像上设置颜色
        }
    }

    // 保存图像
//    for (int i = 0; i < 5; ++i) {
//        std::size_t pos = inputSavePath.find_last_of(".");  // 查找最后一个点的位置
//        std::string result = inputSavePath.substr(0, pos);  // 获取从开头到最后一个点之前的子字符串
//        std::string file = result +"/" + std::to_string(i) + fileName;
//        file_op::File::MkdirFromFile(file);
//        cv::imwrite(file, topViews[i]);
//    }

    // 显示图像
//    file_op::File::MkdirFromFile(inputSavePath);
//    cv::imwrite(inputSavePath, topView);
    cv::imshow("res",topView);
    cv::waitKey(10000);
}



int main() {

    // 读取图像
    std::vector<SyncDataFile> dataset;
    std::string inputDir = "/media/xin/data1/data/parker_data/2022_08_22/louti/data_2023_0822_2"; //数据集路径
    psl::CameraMoudleParam param;
    std::string cameraConfigFile = inputDir + "/config.yaml"; //相机配置文件路径
    GetCameraConfig(cameraConfigFile, param);  // 获取相机配置数据

    const std::string parkerDir = "/home/xin/zhang/c_project/tof/tof_label/config/param_parker.yaml"; //parker配置文件路径
    ConfigParam configParam;
    GetParkerConfig(parkerDir, configParam);  //获取parker配置数据
    bool binocular = GetData(inputDir, dataset, true);

    const size_t size = dataset.size();

    for (size_t i = 0; i < size; ++i) {
        SyncDataFile item = dataset.at(i);
//        item.imageLeft = "/media/xin/data1/data/parker_data/2022_08_22/louti/data_2023_0822_2/20210223_1355/cam0/04_1614045299962078.jpg";
        std::string imageLeftPath(item.imageLeft);
//        std::cout << "item.imageLeft: " << item.imageLeft << ", item.imageCam0: " << item.imageCam0 << std::endl;
        auto imageLeft = cv::imread(imageLeftPath, cv::IMREAD_GRAYSCALE);
        std::cout << "left image read: " << imageLeftPath << std::endl;
        std::string imageRightPath;
        cv::Mat imageRight;

        if (imageLeft.empty()) {
            ERROR_PRINT("empty data in file <" + imageLeftPath + ">");
            continue;
        }

        if (binocular) {
            imageRightPath = item.imageRight;
            imageRight = cv::imread(imageRightPath, cv::IMREAD_GRAYSCALE);
//            std::cout << "right image read: " << imageRightPath << std::endl;

            if (imageRight.empty()) {
                ERROR_PRINT("empty data in file <" + imageRightPath + ">");
                continue;
            }
        }

        // 从配置文件中读取tof数据
        psl::TofDepthData tof;
        std::vector<Eigen::Vector3d> points;
        std::string tofpath = item.tof; //tof配置文件路径
//        std::string tofpath = "/media/xin/data1/data/parker_data/2022_08_22/louti/data_2023_0822_2/20210223_1355/tof/04_1614045299894440_tof.yaml"; //tof配置文件路径
        GetTof(tofpath, tof);
        GetTofPoints(tof, points, configParam.structure.tofAngle); // tof点坐标转换

        // remap操作
        cv::Mat remapX = cv::Mat();
        cv::Mat remapY = cv::Mat();
        ReadPara(param._left_camera.at(RESOLUTION), remapX, remapY);
        ReadPara(param._right_camera.at(RESOLUTION), remapX, remapY);
        const CameraType letftype = LEFT;
        const CameraType righttype = RIGHT;
        Remap(letftype, imageLeft,remapX,remapY);
        Remap(righttype, imageRight,remapX,remapY);

        std::vector<Eigen::Vector3d> pointsSelected;
        std::vector<cv::Point> imagePoints;
        std::vector<int> depthValues;
        const double *p = param._left_camera.at(RESOLUTION)._P;
        Eigen::Matrix<double, 3, 4> P = Eigen::Matrix<double, 3, 4>::Zero();
        P << p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11];
        std::string fileName = GetFileNameFromPath(item.imageLeft);
        std::string lastTwoParts = getLastTwoPathParts(inputDir);
//        std::string topViewPath = "/media/xin/data1/test_data/tof_test/louti_data_2023_0822_2_new/";
        std::string topViewPath = "/media/xin/data1/test_data/tof_test/louti_data_2023_0822_2/tof_with_desk_select/";
        std::string ImagePath = topViewPath + lastTwoParts + item.imageCam0;
        BoxInfo box;
        TofPointsInImage(points,imagePoints,pointsSelected,configParam.structure.leftcamera2tof,P,imageLeft,fileName,box,ImagePath,true);
        drawTopView(ImagePath,pointsSelected,fileName);   //绘制tof文件点

//        drawTopView(ImagePath,points,fileName);   //绘制tof文件点

        // 转3D点
//        TofPointsInImage2DepthValues(pointsSelected,depthValues);
//        camera_fx = param._right_camera[RESOLUTION]._P[0];
//        camera_fy= param._right_camera[RESOLUTION]._P[5];
//        camera_cx = param._right_camera[RESOLUTION]._P[2];
//        camera_cy = param._right_camera[RESOLUTION]._P[6];
//        std::vector<Eigen::Vector3d> cloudPoints;
//        getPointCloud(imageLeft,imagePoints, depthValues,cloudPoints, true);
        // 遍历并打印点云中的点坐标
//        for (const Eigen::Vector3d& point : cloudPoints) {
//            std::cout << "Point coordinates: (" << point.x() << ", " << point.y() << ", " << point.z() << ")" << std::endl;
//        }
//        drawTopView(ImagePath,cloudPoints,fileName);   //绘制tof文件点
//        drawCloudTopView(ImagePath,cloudPoints,fileName);
    }
    return 0;
}



