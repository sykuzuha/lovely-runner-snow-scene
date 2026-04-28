CXX = /usr/bin/clang++
CC = /usr/bin/clang
ARCH = -arch arm64

INCLUDES = -Iinclude -I/opt/homebrew/include
CXXFLAGS = $(ARCH) -std=c++17 -Wall $(INCLUDES)
CFLAGS = $(ARCH) -Wall $(INCLUDES)

GLAD_SRC = src/glad.c
GLAD_OBJ = glad.o

SCENE_SRC = src/main.cpp src/shader.cpp src/audio_player.mm
SCENE_OUT = cat_scene

FUR_SRC = src/main_fur.cpp
FUR_OUT = cat_fur

COMMON_LIBS = $(ARCH) -L/opt/homebrew/lib \
	-framework OpenGL -framework Cocoa \
	-framework IOKit -framework CoreVideo \
	-framework Foundation -framework AVFoundation
SCENE_LIBS = $(COMMON_LIBS) -lglfw -lassimp -lpng
FUR_LIBS = $(COMMON_LIBS) -lglfw

.PHONY: all run fur run_fur clean

all: $(SCENE_OUT)

$(GLAD_OBJ): $(GLAD_SRC)
	$(CC) $(CFLAGS) -c $(GLAD_SRC) -o $(GLAD_OBJ)

$(SCENE_OUT): $(GLAD_OBJ) $(SCENE_SRC)
	$(CXX) $(CXXFLAGS) $(SCENE_SRC) $(GLAD_OBJ) -o $(SCENE_OUT) $(SCENE_LIBS)

run: $(SCENE_OUT)
	./$(SCENE_OUT)

fur: $(FUR_OUT)

$(FUR_OUT): $(GLAD_OBJ) $(FUR_SRC)
	$(CXX) $(CXXFLAGS) $(FUR_SRC) $(GLAD_OBJ) -o $(FUR_OUT) $(FUR_LIBS)

run_fur: $(FUR_OUT)
	./$(FUR_OUT) assets/im_sol_arm_out.glb

clean:
	rm -f $(SCENE_OUT) $(FUR_OUT) $(GLAD_OBJ)
