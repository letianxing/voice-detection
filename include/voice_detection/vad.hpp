#pragma once

#include "voice_detection/types.hpp"

#include <vector>

namespace voice_detection
{

class EnergyVad
{
public:
  EnergyVad(
    float threshold_db = 8.0F,
    float min_rms_dbfs = -52.0F,
    int calibration_frames = 0);

  [[nodiscard]] VoiceActivity process(const std::vector<float> & mono);

private:
  float threshold_db_{8.0F};
  float min_rms_dbfs_{-52.0F};
  float noise_alpha_{0.96F};
  float noise_floor_dbfs_{-65.0F};
  int calibration_frames_{0};
  int calibration_seen_{0};
};

}  // namespace voice_detection
