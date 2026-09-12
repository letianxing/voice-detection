#include "voice_detection/tracker.hpp"

#include "voice_detection/math_utils.hpp"

#include <algorithm>
#include <limits>

namespace voice_detection
{

SourceTracker::SourceTracker(float max_angle_delta_deg, std::int64_t max_age_ms)
: max_angle_delta_deg_(max_angle_delta_deg), max_age_ms_(max_age_ms)
{
}

std::string SourceTracker::assign(std::int64_t stamp_ms, const std::optional<float> & azimuth_deg)
{
  expire(stamp_ms);
  if (!azimuth_deg.has_value()) {
    return "voice_mono";
  }

  const float azimuth = normalizeAngleDeg(*azimuth_deg);
  TrackState * best = nullptr;
  float best_delta = std::numeric_limits<float>::max();
  for (auto & track : tracks_) {
    if (!track.azimuth_deg.has_value()) {
      continue;
    }
    const float delta = angularDistanceDeg(azimuth, *track.azimuth_deg);
    if (delta < best_delta) {
      best_delta = delta;
      best = &track;
    }
  }
  if (best != nullptr && best_delta <= max_angle_delta_deg_) {
    best->azimuth_deg = azimuth;
    best->last_seen_ms = stamp_ms;
    return best->track_id;
  }

  TrackState track;
  track.track_id = "voice_" + std::to_string(next_id_++);
  track.azimuth_deg = azimuth;
  track.last_seen_ms = stamp_ms;
  tracks_.push_back(track);
  return track.track_id;
}

std::vector<std::string> SourceTracker::trackedIds(std::int64_t stamp_ms)
{
  expire(stamp_ms);
  std::vector<std::string> ids;
  ids.reserve(tracks_.size());
  for (const auto & track : tracks_) {
    ids.push_back(track.track_id);
  }
  return ids;
}

void SourceTracker::expire(std::int64_t stamp_ms)
{
  tracks_.erase(
    std::remove_if(
      tracks_.begin(), tracks_.end(),
      [this, stamp_ms](const TrackState & track) {
        return stamp_ms - track.last_seen_ms > max_age_ms_;
      }),
    tracks_.end());
}

}  // namespace voice_detection
