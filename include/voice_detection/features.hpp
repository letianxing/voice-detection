#pragma once

#include "voice_detection/types.hpp"

#include <vector>

namespace voice_detection
{

AudioFeatures extractFeatures(const std::vector<float> & mono, int sample_rate_hz);

}  // namespace voice_detection

