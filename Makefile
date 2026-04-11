CXX = /usr/bin/clang++
CC = /usr/bin/clang

ARCH = -arch arm64

CXXFLAGS = $(ARCH) -std=c++17 -Wall -Iinclude -I/opt/homebrew/include
CFLAGS = $(ARCH) -Wall -Iinclude -I/opt/homebrew/include

CPP_SRC = src/main.cpp
C_SRC = src/camera.cpp

OUT = cat_scene

LIBS = $(ARCH) -L/opt/homebrew/lib -lglfw -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo

all:
	$(CC) $(CFLAGS) -x c -c $(C_SRC) -o glad.o
	$(CXX) $(CXXFLAGS) $(CPP_SRC) glad.o -o $(OUT) $(LIBS)

run: all
	./$(OUT)

clean:
	rm -f $(OUT) glad.o
