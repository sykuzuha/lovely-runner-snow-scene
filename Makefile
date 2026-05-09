CXX = /usr/bin/clang++
CC  = /usr/bin/clang
ARCH = -arch arm64

CXXFLAGS = $(ARCH) -std=c++17 -Wall -Iinclude -I/opt/homebrew/include
CFLAGS   = $(ARCH) -Wall -Iinclude -I/opt/homebrew/include

LIBS = $(ARCH) -L/opt/homebrew/lib -lglfw \
	-framework OpenGL -framework Cocoa \
	-framework IOKit -framework CoreVideo \
	-framework AVFoundation

TARGET = cat_scene
OBJS = glad.o audio_player.o main.o

.PHONY: all run clean

all: $(TARGET)

glad.o: src/glad.c
	$(CC) $(CFLAGS) -c src/glad.c -o glad.o

audio_player.o: src/audio_player.mm
	$(CXX) $(CXXFLAGS) -c src/audio_player.mm -o audio_player.o

main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) -c src/main.cpp -o main.o

$(TARGET): $(OBJS)
	$(CXX) $(ARCH) $(OBJS) -o $(TARGET) $(LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
