CXX = /usr/bin/clang++
CC  = /usr/bin/clang
ARCH = -arch arm64

CXXFLAGS = $(ARCH) -std=c++17 -Wall -Iinclude -I/opt/homebrew/include
CFLAGS   = $(ARCH) -Wall -Iinclude -I/opt/homebrew/include

LIBS_FUR = $(ARCH) -L/opt/homebrew/lib -lglfw \
        -framework OpenGL -framework Cocoa \
        -framework IOKit -framework CoreVideo \
        -framework AVFoundation

all: fur

glad.o: src/glad.c
	$(CC) $(CFLAGS) -c src/glad.c -o glad.o

audio_player.o: src/audio_player.mm
	$(CXX) $(CXXFLAGS) -c src/audio_player.mm -o audio_player.o

main_fur.o: src/main_fur.cpp
	$(CXX) $(CXXFLAGS) -c src/main_fur.cpp -o main_fur.o

fur: glad.o audio_player.o main_fur.o
	$(CXX) $(ARCH) main_fur.o glad.o audio_player.o -o cat_fur $(LIBS_FUR)

run: fur
	./cat_fur assets/im_sol_arm_out.glb assets/sunjae.glb

clean:
	rm -f cat_fur glad.o main_fur.o audio_player.o