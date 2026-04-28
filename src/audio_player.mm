#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

#include "audio_player.h"

struct AudioPlayer {
    AVAudioPlayer* player = nil;
};

namespace {

std::string describeError(NSError* error, const std::string& fallback) {
    if (error == nil || error.localizedDescription == nil) {
        return fallback;
    }

    return std::string([[error localizedDescription] UTF8String]);
}

}  // namespace

AudioPlayer* createAudioPlayer(const std::string& path, std::string& errorMessage) {
    @autoreleasepool {
        NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
        if (nsPath == nil) {
            errorMessage = "Invalid audio path.";
            return nullptr;
        }

        NSURL* url = [NSURL fileURLWithPath:nsPath];
        NSError* error = nil;
        AVAudioPlayer* nativePlayer = [[AVAudioPlayer alloc] initWithContentsOfURL:url error:&error];
        if (nativePlayer == nil) {
            errorMessage = describeError(error, "Failed to open audio file.");
            return nullptr;
        }

        nativePlayer.numberOfLoops = -1;
        [nativePlayer prepareToPlay];

        AudioPlayer* player = new AudioPlayer();
        player->player = nativePlayer;
        return player;
    }
}

bool startAudioPlayer(AudioPlayer* player, std::string& errorMessage) {
    if (player == nullptr || player->player == nil) {
        errorMessage = "Audio player was not initialized.";
        return false;
    }

    @autoreleasepool {
        if ([player->player play]) {
            return true;
        }
    }

    errorMessage = "Audio playback could not be started.";
    return false;
}

void destroyAudioPlayer(AudioPlayer* player) {
    if (player == nullptr) {
        return;
    }

    @autoreleasepool {
        if (player->player != nil) {
            [player->player stop];
            [player->player release];
            player->player = nil;
        }
    }

    delete player;
}
