cmake_minimum_required(VERSION 3.16)
project(voice_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(voice_detection_core
  src/beamforming.cpp
  src/doa.cpp
  src/features.cpp
  src/frontend.cpp
  src/profiles.cpp
  src/signal_enhancement.cpp
  src/tracker.cpp
  src/vad.cpp
)

target_include_directories(voice_detection_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(voice_detection_cli src/voice_detection_cli.cpp)
target_link_libraries(voice_detection_cli PRIVATE voice_detection_core)

enable_testing()
add_executable(voice_detection_core_test tests/cpp_core_test.cpp)
target_link_libraries(voice_detection_core_test PRIVATE voice_detection_core)
add_test(NAME voice_detection_core_test COMMAND voice_detection_core_test)

install(TARGETS voice_detection_core voice_detection_cli
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
