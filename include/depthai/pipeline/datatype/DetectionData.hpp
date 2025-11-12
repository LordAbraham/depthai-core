#pragma once

#include <depthai/pipeline/datatype/Buffer.hpp>
#include <vector>
#include <array>

namespace dai {

/**
 * @brief Single detection result
 * Contains bounding box, confidence score, class label, and optional keypoints
 */
struct Detection {
    float score = 0.0f; // Confidence score
    int label = 0;      // Class label
    std::array<float, 4> bbox; // Bounding box [xmin, ymin, xmax, ymax] normalized [0.0, 1.0]
    std::vector<float> keypoints;     // Flattened keypoints [x0, y0, x1, y1, ...]
};

DEPTHAI_SERIALIZE_EXT(Detection, score, label, bbox, keypoints);

/**
 * DetectionData message. Carries YOLO detection results.
 */
class DetectionData : public Buffer {
   public:
    /**
     * Construct DetectionData message.
     */
    DetectionData();
    
    /**
     * Construct DetectionData message with detections.
     * @param detections Vector of detection results
     */
    DetectionData(const std::vector<Detection>& detections);

    virtual ~DetectionData();

    /// Vector of detections
    std::vector<Detection> detections;
    

    void serialize(std::vector<std::uint8_t>& metadata, DatatypeEnum& datatype) const override;

    /**
     * Get number of detections
     */
    size_t getNumDetections() const { return detections.size(); }
    
    /**
     * Get detections with exactly N keypoints
     * @param n_keypoints Number of keypoints to filter by
     * @return Vector of filtered detections
     */
    std::vector<Detection> getDetectionsWithKeypoints(size_t n_keypoints) const;
    
    /**
     * Get detections by class label
     * @param label Class label to filter by
     * @return Vector of filtered detections
     */
    std::vector<Detection> getDetectionsByLabel(int label) const;
    
    /**
     * Get detection with highest confidence
     * @return Detection with highest score, or empty Detection if no detections
     */
    Detection getBestDetection() const;

    /**
     * Convert detections to flattened format: [kpt0_x, kpt0_y, ..., conf, cls]
     * Useful for serialization or interfacing with other systems
     * @param n_keypoints Number of keypoints to include (0 for bbox only)
     * @return Vector of flattened detection vectors
     */
    std::vector<std::vector<float>> toFlatFormat(size_t n_keypoints = 0) const;

    DEPTHAI_SERIALIZE(DetectionData, Buffer::sequenceNum, Buffer::ts, Buffer::tsDevice, detections);
};

}  // namespace dai
