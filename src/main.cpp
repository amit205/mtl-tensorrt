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

    size_t count = fn.size(); //number of png files in images folder
    std::cout << "Number of images in the folder: " << count << std::endl;
    for (size_t i = 0; i < count; i++) {
        auto img = cv::imread(fn[i], cv::IMREAD_GRAYSCALE);
        cv::resize(img, img, cv::Size(320, 240), cv::INTER_LINEAR);
        cv::imshow("Input", img);
        cv::waitKey(1);
        images.push_back(img);
    }
    
    // Discard the first inference time as it takes longer
    //std::vector<cv::Mat> featureVectors_descriptor;
    //std::vector<cv::Mat> featureVectors_detector;
    std::vector<cv::Mat> featureVectors_segmentation;

    for (size_t i = 0; i < count; ++i) {
        //featureVectors_descriptor.clear();
        //featureVectors_detector.clear();
        featureVectors_segmentation.clear();
        batch_images.clear();
        batch_images.push_back(images[i]);
        //succ = engine_semanticKeypoints.runInference(batch_images, featureVectors_descriptor, featureVectors_detector, featureVectors_segmentation);
        succ = engine_semanticKeypoints.runInference(batch_images, featureVectors_segmentation);
        if (!succ) {
            throw std::runtime_error("Unable to run semanticKeypoints inference.");
        }
        //std::cout << featureVectors_segmentation[0].size<<std::endl;
        //std::cout << featureVectors_descriptor[0].size << std::endl;
        //std::cout << featureVectors_detector[0].size<<std::endl;

        cv::Mat segImg(240, 320, CV_64F, cv::Scalar(0.0));
        float* input = (float*)(featureVectors_segmentation[0].data);
        
        for (int index = 0; index < featureVectors_segmentation[0].size[0] * featureVectors_segmentation[0].size[1] * featureVectors_segmentation[0].size[2]; index++) {
            //std::cout << input[index] << std::endl;
        }
        for (int r = 0; r < 240; r++)
            for (int c = 0; c < 320; c++)
                segImg.at<uint8_t>(r, c) = 255*(static_cast<float_t>(input[(r * 320) + c]));
                //segImg.at<uint8_t>(r, c) = 255*(*(reinterpret_cast<float*>(featureVectors_segmentation[0].data) + r * 320 + c));
        cv::imwrite("/root/home/docker/kitti-odometry-gray/sequences/00/seg_0/" + std::to_string(i) + ".jpg", segImg);
        seg_images.push_back(segImg);

        cv::imshow("Segmentation Image", segImg);
        //std::cout << segImg << std::endl;
        cv::waitKey(1);
        /*
        input = (float*)(featureVectors_descriptor[0].data);
        for (int index = 0; index < featureVectors_descriptor[0].size[0] * featureVectors_descriptor[0].size[1] * featureVectors_descriptor[0].size[2]; index++) {
            //std::cout << input[index] << std::endl;
        }

        input = (float*)(featureVectors_detector[0].data);
        for (int index = 0; index < featureVectors_detector[0].size[0] * featureVectors_detector[0].size[1] * featureVectors_detector[0].size[2]; index++) {
            //std::cout << input[index] << std::endl;
        }*/
        /*
        size_t numIterations = 100;
            auto t1 = Clock::now();
        for (size_t i = 0; i < numIterations; ++i) {
            featureVectors_descriptor.clear();
            featureVectors_detector.clear();
            featureVectors_segmentation.clear();
            engine_semanticKeypoints.runInference(batch_images, featureVectors_descriptor, featureVectors_detector, featureVectors_segmentation);
        }
        auto t2 = Clock::now();
        double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        std::cout << "Success! Average time per inference: " << totalTime / numIterations / static_cast<float>(batch_images.size()) <<
        " ms, for batch size of: " << batch_images.size() << std::endl;
        cv::waitKey(0);
        */
    }
        return 0;
}
