#include "depthai/pipeline/datatype/DetectionData.hpp"
#include <depthai/utility/Serialization.hpp>
#include <algorithm>
#include <limits>

namespace dai {

DetectionData::DetectionData() : imageWidth(0), imageHeight(0) {}

DetectionData::DetectionData(const std::vector<Detection>& detections) 
    : detections(detections), imageWidth(0), imageHeight(0) {}

DetectionData::DetectionData(const std::vector<Detection>& detections, int width, int height)
    : detections(detections), imageWidth(width), imageHeight(height) {}

DetectionData::~DetectionData() = default;

void DetectionData::serialize(std::vector<std::uint8_t>& metadata, DatatypeEnum& datatype) const {
    metadata = utility::serialize(*this);
    datatype = DatatypeEnum::DetectionData;  // Use Buffer type for custom messages
}

std::vector<Detection> DetectionData::getDetectionsWithKeypoints(size_t n_keypoints) const {
    std::vector<Detection> filtered;
    filtered.reserve(detections.size());
    
    for (const auto& det : detections) {
        // keypoints are flattened [x0, y0, x1, y1, ...], so count is size/2
        if (det.keypoints.size() / 2 == n_keypoints) {
            filtered.push_back(det);
        }
    }
    
    return filtered;
}

std::vector<Detection> DetectionData::getDetectionsByLabel(int label) const {
    std::vector<Detection> filtered;
    filtered.reserve(detections.size());
    
    for (const auto& det : detections) {
        if (det.label == label) {
            filtered.push_back(det);
        }
    }
    
    return filtered;
}

Detection DetectionData::getBestDetection() const {
    if (detections.empty()) {
        return Detection{};
    }
    
    auto it = std::max_element(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.score < b.score;
        });
    
    return *it;
}

std::vector<std::vector<float>> DetectionData::toFlatFormat(size_t n_keypoints) const {
    std::vector<std::vector<float>> output;
    output.reserve(detections.size());
    
    for (const auto& det : detections) {
        std::vector<float> row;
        
        // Add bounding box coordinates [xmin, ymin, xmax, ymax]
        row.reserve(4 + (n_keypoints * 2) + 2);  // bbox + keypoints + conf + cls
        row.insert(row.end(), det.bbox.begin(), det.bbox.end());
        
        // Add keypoints if requested
        if (n_keypoints > 0) {
            size_t available_kpts = det.keypoints.size() / 2;  // keypoints are flattened [x0,y0,x1,y1,...]
            size_t kpts_to_add = std::min(n_keypoints, available_kpts);
            
            for (size_t i = 0; i < kpts_to_add; ++i) {
                row.push_back(det.keypoints[i * 2]);      // x coordinate
                row.push_back(det.keypoints[i * 2 + 1]);  // y coordinate
            }
            
            // Pad with zeros if fewer keypoints than requested
            for (size_t i = kpts_to_add; i < n_keypoints; ++i) {
                row.push_back(0.0f);
                row.push_back(0.0f);
            }
        }
        
        // Add confidence and class
        row.push_back(det.score);  // conf
        row.push_back(static_cast<float>(det.label));  // cls
        
        output.push_back(std::move(row));
    }
    
    return output;
}

}  // namespace dai
