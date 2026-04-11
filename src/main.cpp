#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Particle {
    float x;
    float y;
    float speed;
    float drift;
    float phase;
    float size;
};

struct ParticleVertex {
    float x;
    float y;
    float size;
};

constexpr int kWindowWidth = 960;
constexpr int kWindowHeight = 640;
constexpr std::size_t kParticleCount = 900;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    (void)window;
    glViewport(0, 0, width, height);
}

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLint logLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        std::string log(static_cast<std::size_t>(logLength), '\0');
        glGetShaderInfoLog(shader, logLength, nullptr, log.data());
        std::cerr << "Shader compilation failed:\n" << log << '\n';
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

GLuint createProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);
    if (vertexShader == 0 || fragmentShader == 0) {
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLint logLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
        std::string log(static_cast<std::size_t>(logLength), '\0');
        glGetProgramInfoLog(program, logLength, nullptr, log.data());
        std::cerr << "Program linking failed:\n" << log << '\n';
        glDeleteProgram(program);
        program = 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

Particle makeParticle(std::mt19937& rng, bool spawnAtTop) {
    std::uniform_real_distribution<float> xDist(-1.05f, 1.05f);
    std::uniform_real_distribution<float> yDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> speedDist(0.22f, 0.62f);
    std::uniform_real_distribution<float> driftDist(0.5f, 1.8f);
    std::uniform_real_distribution<float> phaseDist(0.0f, 6.28318f);
    std::uniform_real_distribution<float> sizeDist(4.0f, 9.0f);
    std::uniform_real_distribution<float> topOffsetDist(0.0f, 0.35f);

    Particle particle{};
    particle.x = xDist(rng);
    particle.y = spawnAtTop ? 1.05f + topOffsetDist(rng) : yDist(rng);
    particle.speed = speedDist(rng);
    particle.drift = driftDist(rng);
    particle.phase = phaseDist(rng);
    particle.size = sizeDist(rng);
    return particle;
}

}  // namespace

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(kWindowWidth, kWindowHeight, "Lovely Runner Snow Scene", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glViewport(0, 0, kWindowWidth, kWindowHeight);

    const char* particleVertexShader = R"GLSL(
        #version 330 core
        layout (location = 0) in vec2 aPosition;
        layout (location = 1) in float aSize;

        void main() {
            gl_Position = vec4(aPosition, 0.0, 1.0);
            gl_PointSize = aSize;
        }
    )GLSL";

    const char* particleFragmentShader = R"GLSL(
        #version 330 core
        out vec4 FragColor;

        void main() {
            vec2 point = gl_PointCoord - vec2(0.5);
            float distanceFromCenter = length(point);
            if (distanceFromCenter > 0.5) {
                discard;
            }

            float alpha = smoothstep(0.5, 0.0, distanceFromCenter);
            vec3 color = vec3(0.95, 0.97, 1.0);
            FragColor = vec4(color, 0.95 * alpha);
        }
    )GLSL";

    GLuint particleProgram = createProgram(particleVertexShader, particleFragmentShader);
    if (particleProgram == 0) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    GLuint vao = 0;
    GLuint vbo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    std::vector<Particle> particles;
    particles.reserve(kParticleCount);

    std::random_device rd;
    std::mt19937 rng(rd());
    for (std::size_t i = 0; i < kParticleCount; ++i) {
        particles.push_back(makeParticle(rng, false));
    }

    std::vector<ParticleVertex> vertices(kParticleCount);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(ParticleVertex)), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), reinterpret_cast<void*>(offsetof(ParticleVertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), reinterpret_cast<void*>(offsetof(ParticleVertex, size)));
    glEnableVertexAttribArray(1);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    float lastTime = static_cast<float>(glfwGetTime());

    while (!glfwWindowShouldClose(window)) {
        const float currentTime = static_cast<float>(glfwGetTime());
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        if (deltaTime > 0.033f) {
            deltaTime = 0.033f;
        }

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        for (std::size_t i = 0; i < particles.size(); ++i) {
            Particle& particle = particles[i];
            particle.y -= particle.speed * deltaTime;
            particle.x += std::sin(currentTime * particle.drift + particle.phase) * 0.16f * deltaTime;

            if (particle.y < -1.1f || particle.x < -1.15f || particle.x > 1.15f) {
                particle = makeParticle(rng, true);
            }

            vertices[i] = {particle.x, particle.y, particle.size};
        }

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(vertices.size() * sizeof(ParticleVertex)), vertices.data());

        glClearColor(0.06f, 0.10f, 0.18f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(particleProgram);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vertices.size()));

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(particleProgram);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
