#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

// window resize callback
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main() {
    // 1. Initialize GLFW
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW\n";
        return -1;
    }

    // Set OpenGL version (Mac needs this)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // 2. Create window
    GLFWwindow* window = glfwCreateWindow(800, 600, "Lovely Runner Snow Scene", NULL, NULL);
    if (!window) {
        std::cout << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // 3. Load OpenGL functions (GLAD)
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD\n";
        return -1;
    }

    // Set viewport
    glViewport(0, 0, 800, 600);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // 4. Render loop
    while (!glfwWindowShouldClose(window)) {
        // Input (press ESC to close)
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // Clear screen (this is what you’ll see)
        glClearColor(0.1f, 0.1f, 0.2f, 1.0f); // dark blue-ish
        glClear(GL_COLOR_BUFFER_BIT);

        // Swap buffers + poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 5. Cleanup
    glfwTerminate();
    return 0;
}