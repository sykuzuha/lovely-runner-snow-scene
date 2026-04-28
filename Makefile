CXX = /usr/bin/clang++
CC  = /usr/bin/clang
ARCH = -arch arm64
CXXFLAGS = $(ARCH) -std=c++17 -Wall -Iinclude -I/opt/homebrew/include
CFLAGS   = $(ARCH) -Wall -Iinclude -I/opt/homebrew/include

CPP_SRC = src/main.cpp src/shader.cpp
CPP_OBJ = main.o shader.o
C_SRC   = src/glad.c
OUT = cat_scene

LIBS = $(ARCH) -L/opt/homebrew/lib -lglfw \
        -framework OpenGL -framework Cocoa \
        -framework IOKit -framework CoreVideo

all: glad.o $(CPP_OBJ)
	$(CXX) $(ARCH) $(CPP_OBJ) glad.o -o $(OUT) $(LIBS)

glad.o: $(C_SRC)
	$(CC) $(CFLAGS) -c $(C_SRC) -o glad.o

main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) -c src/main.cpp -o main.o

shader.o: src/shader.cpp
	$(CXX) $(CXXFLAGS) -c src/shader.cpp -o shader.o

fur: glad.o
	$(CXX) $(CXXFLAGS) -c src/main_fur.cpp -o main_fur.o
	$(CXX) $(ARCH) main_fur.o glad.o -o cat_fur $(LIBS)

run: all
	./$(OUT)

run_fur: fur
	./cat_fur assets/im_sol_arm_out.glb

clean:
	rm -f $(OUT) cat_fur glad.o $(CPP_OBJ) main_fur.o