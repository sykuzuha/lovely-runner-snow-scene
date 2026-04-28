#pragma once

#include <string>

struct AudioPlayer;

AudioPlayer* createAudioPlayer(const std::string& path, std::string& errorMessage);
bool startAudioPlayer(AudioPlayer* player, std::string& errorMessage);
void destroyAudioPlayer(AudioPlayer* player);
