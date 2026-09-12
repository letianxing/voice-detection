#pragma once

#include "voice_detection/types.hpp"

#include <string>
#include <vector>

namespace voice_detection
{

std::vector<MicrophoneProfile> defaultProfiles();
MicrophoneProfile getDefaultProfile(const std::string & id);

}  // namespace voice_detection

