CXX = /usr/bin/clang++
CC  = /usr/bin/clang
ARCH = -arch arm64
CXXFLAGS = $(ARCH) -std=c++17 -Wall -Iinclude -I/opt/homebrew/include
CFLAGS = $(ARCH) -Wall -Iinclude -I/opt/homebrew/include

CPP_SRC = src/main.cpp
C_SRC = src/camera.cpp

CFLAGS   = $(ARCH) -Wall -Iinclude -I/opt/homebrew/include
CPP_SRC = src/main.cpp src/camera.cpp src/shader.cpp
C_SRC   = src/glad.c
OUT = cat_scene
LIBS = $(ARCH) -L/opt/homebrew/lib -lglfw \
		-framework OpenGL -framework Cocoa \
		-framework IOKit -framework CoreVideo

all:
	$(CC) $(CFLAGS) -x c -c $(C_SRC) -o glad.o
	$(CC)  $(CFLAGS)   -c $(C_SRC) -o glad.o
	$(CXX) $(CXXFLAGS) $(CPP_SRC) glad.o -o $(OUT) $(LIBS)

run: all
	./$(OUT)

fur:
	$(CC)  $(CFLAGS)   -c src/glad.c -o glad.o
	$(CXX) $(CXXFLAGS) src/main_fur.cpp glad.o -o cat_fur $(LIBS)

run_fur: fur
	./cat_fur assets/im_sol_arm_out.glb

clean:
	rm -f $(OUT) glad.o
	rm -f $(OUT) cat_fur glad.o
