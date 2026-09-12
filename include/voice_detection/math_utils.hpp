#pragma once

#include <algorithm>
#include <cmath>

namespace voice_detection
{

inline float clamp(float value, float low, float high)
{
  return std::max(low, std::min(value, high));
}

inline float sigmoid(float value)
{
  return 1.0F / (1.0F + std::exp(-value));
}

inline float normalizeAngleDeg(float angle)
{
  float normalized = std::fmod(angle + 180.0F, 360.0F);
  if (normalized < 0.0F) {
    normalized += 360.0F;
  }
  normalized -= 180.0F;
  return normalized == -180.0F ? 180.0F : normalized;
}

inline float angularDistanceDeg(float a, float b)
{
  const float delta = std::abs(normalizeAngleDeg(a - b));
  return std::min(delta, 360.0F - delta);
}

}  // namespace voice_detection

