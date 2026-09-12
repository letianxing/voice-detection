#include "voice_detection/vad.hpp"

#include "voice_detection/math_utils.hpp"

#include <cmath>
#include <numeric>

namespace voice_detection
{

EnergyVad::EnergyVad(float threshold_db, float min_rms_dbfs, int calibration_frames)
: threshold_db_(threshold_db), min_rms_dbfs_(min_rms_dbfs),
  calibration_frames_(std::max(0, calibration_frames))
{
}

VoiceActivity EnergyVad::process(const std::vector<float> & mono)
{
  if (mono.empty()) {
    return {};
  }
  double energy = 0.0;
  for (const float sample : mono) {
    energy += static_cast<double>(sample) * static_cast<double>(sample);
  }
  const float rms = static_cast<float>(std::sqrt((energy / static_cast<double>(mono.size())) + 1e-12));
  const float rms_dbfs = 20.0F * std::log10(std::max(rms, 1e-9F));
  if (calibration_seen_ < calibration_frames_) {
    if (calibration_seen_ == 0) {
      noise_floor_dbfs_ = rms_dbfs;
    } else {
      noise_floor_dbfs_ = std::max(noise_floor_dbfs_, rms_dbfs);
    }
    ++calibration_seen_;
    return VoiceActivity{false, 0.0F, rms_dbfs, noise_floor_dbfs_, 0.0F};
  }
  const float snr_db = rms_dbfs - noise_floor_dbfs_;
  const bool active = rms_dbfs >= min_rms_dbfs_ && snr_db >= threshold_db_;
  if (!active) {
    noise_floor_dbfs_ = (noise_alpha_ * noise_floor_dbfs_) + ((1.0F - noise_alpha_) * rms_dbfs);
  }
  float probability = sigmoid((snr_db - threshold_db_) / 3.0F);
  if (rms_dbfs < min_rms_dbfs_) {
    probability *= 0.4F;
  }
  return VoiceActivity{
    active,
    clamp(probability, 0.0F, 1.0F),
    rms_dbfs,
    noise_floor_dbfs_,
    snr_db};
}

}  // namespace voice_detection
