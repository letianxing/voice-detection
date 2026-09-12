#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace voice_detection
{

class SourceTracker
{
public:
  SourceTracker(float max_angle_delta_deg = 25.0F, std::int64_t max_age_ms = 1200);

  std::string assign(std::int64_t stamp_ms, const std::optional<float> & azimuth_deg);
  std::vector<std::string> trackedIds(std::int64_t stamp_ms);

private:
  struct TrackState
  {
    std::string track_id;
    std::optional<float> azimuth_deg;
    std::int64_t last_seen_ms{0};
  };

  void expire(std::int64_t stamp_ms);

  float max_angle_delta_deg_{25.0F};
  std::int64_t max_age_ms_{1200};
  std::vector<TrackState> tracks_;
  int next_id_{1};
};

}  // namespace voice_detection

