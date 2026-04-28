CXX = /usr/bin/clang++
CC = /usr/bin/clang
ARCH = -arch arm64

INCLUDES = -Iinclude -I/opt/homebrew/include
CXXFLAGS = $(ARCH) -std=c++17 -Wall $(INCLUDES)
CFLAGS = $(ARCH) -Wall $(INCLUDES)

CPP_SRC = src/main.cpp
C_SRC = src/glad.c
OUT = cat_scene
FUR_OUT = cat_fur

LIBS = $(ARCH) -L/opt/homebrew/lib -lglfw -lassimp -lpng \
	-framework OpenGL -framework Cocoa \
	-framework IOKit -framework CoreVideo

all: $(OUT)

glad.o: $(C_SRC)
	$(CC) $(CFLAGS) -c $(C_SRC) -o glad.o

$(OUT): glad.o $(CPP_SRC)
CFLAGS   = $(ARCH) -Wall -Iinclude -I/opt/homebrew/include

CPP_SRC = src/main.cpp src/shader.cpp
CPP_OBJ = main.o shader.o
C_SRC   = src/glad.c
OUT = cat_scene

LIBS = $(ARCH) -L/opt/homebrew/lib -lglfw \
        -framework OpenGL -framework Cocoa \
        -framework IOKit -framework CoreVideo

all:
	$(CC)  $(CFLAGS)   -c $(C_SRC) -o glad.o
	$(CXX) $(CXXFLAGS) $(CPP_SRC) glad.o -o $(OUT) $(LIBS)

run: $(OUT)
	./$(OUT)

fur: glad.o src/main_fur.cpp
	$(CXX) $(CXXFLAGS) src/main_fur.cpp glad.o -o $(FUR_OUT) $(LIBS)

run_fur: fur
	./$(FUR_OUT) assets/im_sol_arm_out.glb

clean:
	rm -f $(OUT) $(FUR_OUT) glad.o
	rm -f $(OUT) cat_fur glad.o
