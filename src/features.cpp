#include "voice_detection/features.hpp"

#include "voice_detection/math_utils.hpp"

#include <cmath>

namespace voice_detection
{

AudioFeatures extractFeatures(const std::vector<float> & mono, int sample_rate_hz)
{
  AudioFeatures features;
  if (mono.empty()) {
    return features;
  }

  double energy = 0.0;
  int crossings = 0;
  for (std::size_t i = 0; i < mono.size(); ++i) {
    energy += static_cast<double>(mono[i]) * static_cast<double>(mono[i]);
    if (i > 0 && (mono[i] < 0.0F) != (mono[i - 1] < 0.0F)) {
      ++crossings;
    }
  }
  features.rms = static_cast<float>(std::sqrt(energy / static_cast<double>(mono.size())));
  features.zcr = mono.size() > 1 ? static_cast<float>(crossings) / static_cast<float>(mono.size() - 1) : 0.0F;

  const int min_lag = std::max(1, sample_rate_hz / 500);
  const int max_lag = std::min(static_cast<int>(mono.size()) - 1, sample_rate_hz / 70);
  if (max_lag > min_lag && features.rms > 1e-5F) {
    double best = 0.0;
    int best_lag = 0;
    for (int lag = min_lag; lag < max_lag; ++lag) {
      double corr = 0.0;
      for (std::size_t i = 0; i + static_cast<std::size_t>(lag) < mono.size(); ++i) {
        corr += static_cast<double>(mono[i]) * static_cast<double>(mono[i + static_cast<std::size_t>(lag)]);
      }
      if (corr > best) {
        best = corr;
        best_lag = lag;
      }
    }
    if (best_lag > 0) {
      const float pitch_hz = static_cast<float>(sample_rate_hz) / static_cast<float>(best_lag);
      features.pitch = clamp(pitch_hz / 500.0F, 0.0F, 1.0F);
      features.hnr = clamp(static_cast<float>(best / (energy + 1e-9)), 0.0F, 1.0F);
    }
  }
  return features;
}

}  // namespace voice_detection

