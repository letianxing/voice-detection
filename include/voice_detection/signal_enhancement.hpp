#pragma once

#include <optional>
#include <vector>

namespace voice_detection
{

class AdaptiveNoiseSuppressor
{
public:
  AdaptiveNoiseSuppressor(float noise_alpha = 0.98F, float max_reduction = 0.35F);

  std::vector<float> process(const std::vector<float> & mono, bool speech_active_hint);

private:
  float noise_alpha_{0.98F};
  float max_reduction_{0.35F};
  float noise_rms_{0.003F};
};

class LinearEchoCanceller
{
public:
  LinearEchoCanceller(float smoothing = 0.75F, float max_gain = 2.0F);

  std::vector<float> process(
    const std::vector<float> & microphone,
    const std::optional<std::vector<float>> & playback_reference);

  float echoGain() const {return echo_gain_;}

private:
  float smoothing_{0.75F};
  float max_gain_{2.0F};
  float echo_gain_{0.0F};
};

std::vector<float> removeDcPerChannel(const std::vector<float> & samples, int channels);

}  // namespace voice_detection
