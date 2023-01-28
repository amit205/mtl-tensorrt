#include "engine.h"
#include <opencv2/opencv.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;


int main() {
    Options options;
    options.optBatchSizes = {2, 4, 6};
    Engine engine_semanticKeypoints(options);

    // TODO: Specify your model here.
    // Must specify a dynamic batch size when exporting the model from onnx.
    const std::string onnxModelpath_semanticKeypoints = "../semantic_model.onnx";
    bool succ = engine_semanticKeypoints.build(onnxModelpath_semanticKeypoints);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT semanticKeypoints engine.");
    }
    succ = engine_semanticKeypoints.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT semanticKeypoints engine.");
    }

    const size_t batchSize = 1;
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> seg_images;
    std::vector<cv::Mat> batch_images;
    std::vector<cv::String> fn;
    cv::glob("/root/home/docker/kitti-odometry-gray/sequences/00/image_0/*.png", fn, false);

    //size_t count = fn.size(); //number of png files in images folder
    size_t count = 1000;
    std::cout << "Number of images in the folder: " << count << std::endl;
    for (size_t i = 0; i < count; i++) {
        auto img = cv::imread(fn[i], cv::IMREAD_GRAYSCALE);
        cv::resize(img, img, cv::Size(320, 240), cv::INTER_CUBIC);
        cv::imshow("Input", img);
        cv::waitKey(1);
        images.push_back(img);
    }
    
    // Discard the first inference time as it takes longer
    std::vector<cv::Mat> featureVectors_segmentation;

    for (size_t i = 0; i < count; ++i) {
        featureVectors_segmentation.clear();
        batch_images.clear();
        batch_images.push_back(images[i]);
        succ = engine_semanticKeypoints.runInference(batch_images, featureVectors_segmentation);
        if (!succ) {
            throw std::runtime_error("Unable to run semanticKeypoints inference.");
        }

        cv::Mat segImg(240, 320, CV_32FC1, featureVectors_segmentation[0].data);
        cv::imwrite("/root/home/docker/kitti-odometry-gray/sequences/00/seg_0/" + std::to_string(i) + ".jpg", 255.f*(segImg>=0.5));
        seg_images.push_back(segImg);

        cv::imshow("Segmentation Image", segImg);
        cv::waitKey(1);
    }
        return 0;
}
