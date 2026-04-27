#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

const int W = 1200;
const int H = 900;

const int STRANDS_PER_FACE = 20;
const int SEGMENTS         = 14;
const float BASE_LENGTH    = 0.036f;
const float CURL_AMOUNT    = 0.12f;
const float LENGTH_VARIANCE = 0.2f;
const float CURL_FREQ = 1.0f;
const int RANDOM_SEED = 42;

float orbitYaw = 180.0f;
float orbitPitch = 6.0f;
float orbitDist = 4.5f;
glm::vec3 orbitTarget(0.0f);
double lastX = W / 2.0;
double lastY = H / 2.0;
bool mouseDown = false;

void framebuffer_size_callback(GLFWwindow*, int w, int h) { glViewport(0, 0, w, h); }

void mouse_button_callback(GLFWwindow*, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) mouseDown = (action == GLFW_PRESS);
}

void cursor_callback(GLFWwindow*, double xpos, double ypos) {
    float dx = static_cast<float>(xpos - lastX);
    float dy = static_cast<float>(ypos - lastY);
    lastX = xpos;
    lastY = ypos;
    if (!mouseDown) return;
    orbitYaw += dx * 0.4f;
    orbitPitch = glm::clamp(orbitPitch - dy * 0.4f, -89.0f, 89.0f);
}

void scroll_callback(GLFWwindow*, double, double yoff) {
    orbitDist = glm::clamp(orbitDist - static_cast<float>(yoff) * 0.2f, 0.35f, 20.0f);
}

GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[1024];
        glGetShaderInfoLog(shader, sizeof(buf), nullptr, buf);
        std::cerr << "Shader error: " << buf << "\n";
    }
    return shader;
}

GLuint makeProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[1024];
        glGetProgramInfoLog(p, sizeof(buf), nullptr, buf);
        std::cerr << "Program link error: " << buf << "\n";
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

struct ImageData {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<unsigned char> pixels;
};

struct MaterialData {
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    int textureIndex = -1;
    bool doubleSided = false;
};

struct SubMesh {
    std::string name;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::ivec3> faces;
    int materialIndex = -1;
};

struct SceneData {
    std::vector<SubMesh> submeshes;
    std::vector<MaterialData> materials;
    std::vector<ImageData> images;
    glm::vec3 boundsMin = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3 boundsMax = glm::vec3(-std::numeric_limits<float>::max());
};

glm::mat4 nodeLocalMatrix(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        return glm::make_mat4(node.matrix.data());
    }

    glm::vec3 t(0.0f);
    glm::vec3 s(1.0f);
    glm::quat r(1.0f, 0.0f, 0.0f, 0.0f);
    if (node.translation.size() == 3) {
        t = glm::vec3(
            static_cast<float>(node.translation[0]),
            static_cast<float>(node.translation[1]),
            static_cast<float>(node.translation[2]));
    }
    if (node.scale.size() == 3) {
        s = glm::vec3(
            static_cast<float>(node.scale[0]),
            static_cast<float>(node.scale[1]),
            static_cast<float>(node.scale[2]));
    }
    if (node.rotation.size() == 4) {
        r = glm::quat(
            static_cast<float>(node.rotation[3]),
            static_cast<float>(node.rotation[0]),
            static_cast<float>(node.rotation[1]),
            static_cast<float>(node.rotation[2]));
    }

    return glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(r) *
           glm::scale(glm::mat4(1.0f), s);
}

size_t componentSizeInBytes(int componentType) {
    switch (componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        case TINYGLTF_COMPONENT_TYPE_BYTE:
            return 1;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        case TINYGLTF_COMPONENT_TYPE_SHORT:
            return 2;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        case TINYGLTF_COMPONENT_TYPE_INT:
        case TINYGLTF_COMPONENT_TYPE_FLOAT:
            return 4;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:
            return 8;
        default:
            return 0;
    }
}

std::vector<glm::vec3> readVec3Accessor(const tinygltf::Model& model, int accessorIndex) {
    const auto& accessor = model.accessors[accessorIndex];
    const auto& view = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[view.buffer];
    const unsigned char* data = buffer.data.data() + view.byteOffset + accessor.byteOffset;
    const size_t stride = accessor.ByteStride(view) ? accessor.ByteStride(view) : 3 * sizeof(float);

    std::vector<glm::vec3> out(accessor.count);
    for (size_t i = 0; i < accessor.count; ++i) {
        const float* src = reinterpret_cast<const float*>(data + i * stride);
        out[i] = glm::vec3(src[0], src[1], src[2]);
    }
    return out;
}

std::vector<glm::vec2> readVec2Accessor(const tinygltf::Model& model, int accessorIndex) {
    const auto& accessor = model.accessors[accessorIndex];
    const auto& view = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[view.buffer];
    const unsigned char* data = buffer.data.data() + view.byteOffset + accessor.byteOffset;
    const size_t stride = accessor.ByteStride(view) ? accessor.ByteStride(view) : 2 * sizeof(float);

    std::vector<glm::vec2> out(accessor.count, glm::vec2(0.0f));
    for (size_t i = 0; i < accessor.count; ++i) {
        const float* src = reinterpret_cast<const float*>(data + i * stride);
        out[i] = glm::vec2(src[0], src[1]);
    }
    return out;
}

uint32_t readIndexValue(const unsigned char* ptr, int componentType) {
    switch (componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return *reinterpret_cast<const uint8_t*>(ptr);
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return *reinterpret_cast<const uint16_t*>(ptr);
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return *reinterpret_cast<const uint32_t*>(ptr);
        default:
            return 0;
    }
}

std::vector<glm::ivec3> readIndexAccessor(const tinygltf::Model& model, int accessorIndex) {
    const auto& accessor = model.accessors[accessorIndex];
    const auto& view = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[view.buffer];
    const unsigned char* data = buffer.data.data() + view.byteOffset + accessor.byteOffset;
    const size_t elemSize = componentSizeInBytes(accessor.componentType);
    const size_t stride = accessor.ByteStride(view) ? accessor.ByteStride(view) : elemSize;

    std::vector<glm::ivec3> faces;
    faces.reserve(accessor.count / 3);
    for (size_t i = 0; i + 2 < accessor.count; i += 3) {
        int a = static_cast<int>(readIndexValue(data + i * stride, accessor.componentType));
        int b = static_cast<int>(readIndexValue(data + (i + 1) * stride, accessor.componentType));
        int c = static_cast<int>(readIndexValue(data + (i + 2) * stride, accessor.componentType));
        faces.emplace_back(a, b, c);
    }
    return faces;
}

ImageData decodeImage(const tinygltf::Model& model, const tinygltf::Image& image) {
    ImageData out;
    if (image.bufferView < 0) return out;

    const auto& view = model.bufferViews[image.bufferView];
    const auto& buffer = model.buffers[view.buffer];
    const unsigned char* bytes = buffer.data.data() + view.byteOffset;
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* decoded = stbi_load_from_memory(
        bytes, static_cast<int>(view.byteLength), &width, &height, &channels, 4);
    if (!decoded) return out;

    out.width = width;
    out.height = height;
    out.channels = 4;
    out.pixels.assign(decoded, decoded + width * height * 4);
    stbi_image_free(decoded);
    return out;
}

SceneData loadGLB(const std::string& path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    bool ok = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    if (!warn.empty()) std::cerr << "GLB warning: " << warn << "\n";
    if (!ok) {
        std::cerr << "GLB load error: " << err << "\n";
        return {};
    }

    SceneData scene;
    scene.images.reserve(model.images.size());
    for (const auto& image : model.images) scene.images.push_back(decodeImage(model, image));

    scene.materials.reserve(model.materials.size());
    for (const auto& material : model.materials) {
        MaterialData m;
        m.doubleSided = material.doubleSided;
        if (material.values.count("baseColorFactor")) {
            const auto& col = material.values.at("baseColorFactor").ColorFactor();
            m.baseColorFactor = glm::vec4(
                static_cast<float>(col[0]),
                static_cast<float>(col[1]),
                static_cast<float>(col[2]),
                static_cast<float>(col[3]));
        }
        if (material.pbrMetallicRoughness.baseColorTexture.index >= 0) {
            const int texIndex = material.pbrMetallicRoughness.baseColorTexture.index;
            if (texIndex >= 0 && texIndex < static_cast<int>(model.textures.size())) {
                m.textureIndex = model.textures[texIndex].source;
            }
        }
        scene.materials.push_back(m);
    }
    if (scene.materials.empty()) scene.materials.push_back(MaterialData{});

    std::function<void(int, const glm::mat4&)> visitNode = [&](int nodeIndex, const glm::mat4& parent) {
        const tinygltf::Node& node = model.nodes[nodeIndex];
        glm::mat4 world = parent * nodeLocalMatrix(node);

        if (node.mesh >= 0) {
            const tinygltf::Mesh& mesh = model.meshes[node.mesh];
            glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(world)));

            for (const auto& prim : mesh.primitives) {
                if (!prim.attributes.count("POSITION") || prim.indices < 0) continue;

                SubMesh out;
                out.name = node.name;
                out.materialIndex = prim.material >= 0 ? prim.material : 0;
                auto localPositions = readVec3Accessor(model, prim.attributes.at("POSITION"));
                if (prim.attributes.count("NORMAL")) {
                    out.normals = readVec3Accessor(model, prim.attributes.at("NORMAL"));
                } else {
                    out.normals.assign(localPositions.size(), glm::vec3(0.0f, 1.0f, 0.0f));
                }
                if (prim.attributes.count("TEXCOORD_0")) {
                    out.uvs = readVec2Accessor(model, prim.attributes.at("TEXCOORD_0"));
                } else {
                    out.uvs.assign(localPositions.size(), glm::vec2(0.0f));
                }
                out.faces = readIndexAccessor(model, prim.indices);

                out.positions.reserve(localPositions.size());
                for (size_t i = 0; i < localPositions.size(); ++i) {
                    glm::vec3 p = glm::vec3(world * glm::vec4(localPositions[i], 1.0f));
                    glm::vec3 n = glm::normalize(normalMat * out.normals[i]);
                    out.positions.push_back(p);
                    out.normals[i] = glm::any(glm::isnan(n)) ? glm::vec3(0.0f, 1.0f, 0.0f) : n;
                    scene.boundsMin = glm::min(scene.boundsMin, p);
                    scene.boundsMax = glm::max(scene.boundsMax, p);
                }
                scene.submeshes.push_back(std::move(out));
            }
        }

        for (int child : node.children) visitNode(child, world);
    };

    glm::mat4 orientationFix = glm::mat4(1.0f);

    int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (!model.scenes.empty() && sceneIndex >= 0 && sceneIndex < static_cast<int>(model.scenes.size())) {
        for (int root : model.scenes[sceneIndex].nodes) visitNode(root, orientationFix);
    } else {
        for (int i = 0; i < static_cast<int>(model.nodes.size()); ++i) visitNode(i, orientationFix);
    }

    return scene;
}

glm::vec3 faceNormal(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
    glm::vec3 n = glm::cross(b - a, c - a);
    if (glm::dot(n, n) < 1e-10f) return glm::vec3(0.0f, 1.0f, 0.0f);
    return glm::normalize(n);
}

glm::vec3 makeTangent(const glm::vec3& n) {
    glm::vec3 ref = (std::abs(n.x) < 0.9f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    return glm::normalize(glm::cross(n, ref));
}

glm::vec3 sampleMaterialColor(
    const MaterialData& material, const ImageData* image, const glm::vec2& uv) {
    glm::vec3 color = glm::vec3(material.baseColorFactor);
    if (!image || image->pixels.empty() || image->width <= 0 || image->height <= 0) return color;

    float u = uv.x - std::floor(uv.x);
    float v = uv.y - std::floor(uv.y);

    int x = glm::clamp(static_cast<int>(u * static_cast<float>(image->width - 1)), 0, image->width - 1);
    int y = glm::clamp(static_cast<int>(v * static_cast<float>(image->height - 1)), 0, image->height - 1);
    size_t idx = static_cast<size_t>((y * image->width + x) * 4);

    glm::vec3 texel(
        image->pixels[idx + 0] / 255.0f,
        image->pixels[idx + 1] / 255.0f,
        image->pixels[idx + 2] / 255.0f);
    return color * texel;
}

bool shouldGrowFur(const glm::vec3& color) {
    float maxC = std::max(color.r, std::max(color.g, color.b));
    float minC = std::min(color.r, std::min(color.g, color.b));
    float luminance = glm::dot(color, glm::vec3(0.2126f, 0.7152f, 0.0722f));

    bool darkFeatureLike = luminance < 0.22f;
    return !(darkFeatureLike);
}

std::vector<float> generateFur(
    const SubMesh& mesh,
    const MaterialData& material,
    const ImageData* image,
    bool isHeadMesh) {
    std::mt19937 rng(RANDOM_SEED);
    std::uniform_real_distribution<float> rnd01(0.0f, 1.0f);
    std::uniform_real_distribution<float> rndSym(-1.0f, 1.0f);

    const glm::vec3 up(0.0f, 1.0f, 0.0f);

    glm::vec3 centroid(0.0f);
    for (const auto& p : mesh.positions) centroid += p;
    centroid /= static_cast<float>(std::max<size_t>(mesh.positions.size(), 1));

    std::vector<float> verts;
    verts.reserve(mesh.faces.size() * STRANDS_PER_FACE * SEGMENTS * 14);

    auto softStep = [](float x, float edge0, float edge1) {
        float t = glm::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    };

    for (const auto& face : mesh.faces) {
        const glm::vec3& a = mesh.positions[face.x];
        const glm::vec3& b = mesh.positions[face.y];
        const glm::vec3& c = mesh.positions[face.z];

        glm::vec3 normal = faceNormal(a, b, c);
        glm::vec3 faceCtr = (a + b + c) / 3.0f;
        glm::vec3 outward = faceCtr - centroid;

        if (glm::dot(normal, outward) < -0.1f) continue;

        glm::vec3 tang = makeTangent(normal);
        glm::vec3 bitang = glm::normalize(glm::cross(normal, tang));
        glm::vec3 toFace = glm::normalize(outward);
        float dy = faceCtr.y - centroid.y;

        float regionalLengthMult = 1.0f;
        glm::vec3 grownDir = normal;
        float regionalGravity = 0.04f;
        int strandsThisFace = STRANDS_PER_FACE;

        if (isHeadMesh) {
            glm::vec2 uvCtr = mesh.uvs.empty()
                ? glm::vec2(0.0f)
                : (mesh.uvs[face.x] + mesh.uvs[face.y] + mesh.uvs[face.z]) / 3.0f;
            glm::vec3 faceColor = sampleMaterialColor(material, image, uvCtr);
            float luminance = glm::dot(faceColor, glm::vec3(0.2126f, 0.7152f, 0.0722f));

            if (luminance < 0.08f) continue;

            // faces pointing downward = chin/neck area
            float wNeck = softStep(-normal.y, 0.2f, 0.7f);

            regionalLengthMult = (luminance > 0.72f) ? 0.70f : 0.50f;
            regionalLengthMult = glm::mix(regionalLengthMult, 1.10f, wNeck); // neck gets longer fur
            regionalGravity = glm::mix(0.008f, 0.06f, wNeck); // neck fur droops slightly
            grownDir = glm::normalize(normal * 0.96f + toFace * 0.04f);
            strandsThisFace = (luminance > 0.72f) ? 15 : 10; 
            strandsThisFace = static_cast<int>(strandsThisFace * glm::mix(1.0f, 2.0f, wNeck));
        }
        else {
            if (faceCtr.y < centroid.y - 1.0f) continue;
            if (faceCtr.y > centroid.y + 0.55f) continue;

            float wBack = softStep(dy, 0.15f, 0.40f);
            float wBelly = softStep(-dy, 0.08f, 0.28f);
            float wSide = softStep(std::abs(toFace.x), 0.3f, 0.7f) * (1.0f - wBack * 0.6f);
            float wFront = softStep(toFace.z, 0.0f, 0.6f); 
float wFace = softStep(dy, 0.2f, 0.4f) * softStep(toFace.z, 0.4f, 0.8f); 
            float wHead = softStep(dy, 0.02f, 0.26f);

            bool isLeg = (faceCtr.y < centroid.y - 0.05f) && (std::abs(normal.y) < 0.6f);
            float wLeg = isLeg ? softStep(centroid.y - faceCtr.y, 0.04f, 0.28f) : 0.0f;
            wBelly *= (1.0f - wLeg * 0.4f);

            float blendedW = wBack + wBelly + wSide + wFront + wLeg;

            glm::vec3 flowBack = glm::normalize(glm::vec3(0.0f, -0.15f, 0.30f));
            glm::vec3 flowBelly = glm::normalize(glm::vec3(0.0f, -0.20f, -0.15f));
            glm::vec3 flowSide = glm::normalize(glm::vec3(0.0f, -0.45f, 0.10f));
            glm::vec3 flowFront = glm::normalize(glm::vec3(0.0f, -0.35f, -0.28f));
            glm::vec3 flowLeg = glm::normalize(glm::vec3(toFace.x * 0.4f, -0.35f, toFace.z * 0.4f));
            glm::vec3 flowRump = glm::normalize(glm::vec3(0.0f, -0.25f, 0.22f));

            glm::vec3 blendedFlow = flowBack * wBack + flowBelly * wBelly + flowSide * wSide +
                                    flowFront * wFront + flowLeg * wLeg;
            blendedFlow += flowRump * glm::max(0.0f, 1.0f - blendedW);
            blendedFlow = glm::normalize(blendedFlow);

            regionalLengthMult = 1.05f * wBack + 1.15f * wBelly + 1.00f * wSide +
                     1.20f * wFront + 0.60f * wLeg +  
                     0.95f * glm::max(0.0f, 1.0f - blendedW);
            regionalLengthMult = glm::max(regionalLengthMult, 0.50f);
            regionalLengthMult = glm::mix(regionalLengthMult, 0.75f, glm::clamp(wHead, 0.0f, 1.0f));
            regionalLengthMult = glm::mix(regionalLengthMult, 0.08f, glm::clamp(wFace, 0.0f, 1.0f));

            glm::vec3 flowOnSurface = blendedFlow - glm::dot(blendedFlow, normal) * normal;
            float flowLen = glm::length(flowOnSurface);
            if (flowLen > 0.001f) {
                flowOnSurface = glm::normalize(flowOnSurface);
                grownDir = glm::normalize(normal * 0.40f + flowOnSurface * 0.60f);
            } else {
                grownDir = glm::normalize(normal * 0.15f + tang * 0.85f);
            }

            regionalGravity = 0.55f * wBack + 0.15f * wBelly + 0.45f * wSide +
                              0.40f * wFront + 0.08f * wLeg +
                              0.45f * glm::max(0.0f, 1.0f - blendedW);
            float wSum = wBack + wBelly + wSide + wFront + wLeg +
                         glm::max(0.0f, 1.0f - blendedW);
            if (wSum > 0.001f) regionalGravity /= wSum;
            regionalGravity = glm::clamp(regionalGravity * 0.3f, 0.01f, 0.15f);
            regionalGravity = glm::mix(regionalGravity, 0.05f, glm::clamp(wHead, 0.0f, 1.0f));
            regionalGravity = glm::mix(regionalGravity, 0.06f, glm::clamp(wFace, 0.0f, 1.0f));

            strandsThisFace = std::max(
                1, static_cast<int>(std::lround(STRANDS_PER_FACE * glm::mix(0.70f, 0.25f, wHead))));
            strandsThisFace = std::max(
                0, static_cast<int>(std::lround(static_cast<float>(strandsThisFace) * glm::mix(1.0f, 0.08f, wFace))));

            if (wFace > 0.85f) continue;
        }

        for (int si = 0; si < strandsThisFace; ++si) {
            float r1 = rnd01(rng);
            float r2 = rnd01(rng);
            if (r1 + r2 > 1.0f) {
                r1 = 1.0f - r1;
                r2 = 1.0f - r2;
            }
            float r3 = 1.0f - r1 - r2;

            glm::vec3 origin = r1 * a + r2 * b + r3 * c;
            glm::vec2 uv(0.0f);
            if (!mesh.uvs.empty()) {
                uv = r1 * mesh.uvs[face.x] + r2 * mesh.uvs[face.y] + r3 * mesh.uvs[face.z];
            }
            glm::vec3 furColor = sampleMaterialColor(material, image, uv);
            if (!shouldGrowFur(furColor)) continue;

            float length = BASE_LENGTH * regionalLengthMult *
                           (1.0f + rndSym(rng) * LENGTH_VARIANCE);
            float curlPh = rnd01(rng) * 2.0f * glm::pi<float>();
            float segLen = length / static_cast<float>(SEGMENTS);

            float clumpAngle = rnd01(rng) * 2.0f * glm::pi<float>();
            float clumpStr = 0.25f + rnd01(rng) * 0.20f;
            float leanT = clumpStr * std::cos(clumpAngle) * 0.12f;
            float leanB = clumpStr * std::sin(clumpAngle) * 0.12f;

            glm::vec3 pos = origin;
            for (int s = 0; s < SEGMENTS; ++s) {
                float t0 = static_cast<float>(s) / static_cast<float>(SEGMENTS);
                float t1 = static_cast<float>(s + 1) / static_cast<float>(SEGMENTS);

                glm::vec3 dir = glm::normalize(grownDir + leanT * tang + leanB * bitang);
                float angle = CURL_FREQ * 2.0f * glm::pi<float>() * t0 + curlPh;
                dir += CURL_AMOUNT * (std::cos(angle) * tang + std::sin(angle) * bitang);
                dir -= regionalGravity * t0 * t0 * up;
                dir = glm::normalize(dir);

                glm::vec3 next = pos + dir * segLen;

                verts.push_back(pos.x);
                verts.push_back(pos.y);
                verts.push_back(pos.z);
                verts.push_back(furColor.r);
                verts.push_back(furColor.g);
                verts.push_back(furColor.b);
                verts.push_back(t0);

                verts.push_back(next.x);
                verts.push_back(next.y);
                verts.push_back(next.z);
                verts.push_back(furColor.r);
                verts.push_back(furColor.g);
                verts.push_back(furColor.b);
                verts.push_back(t1);

                pos = next;
            }
        }
    }

    return verts;
}

std::vector<float> flattenMesh(const SubMesh& mesh) {
    std::vector<float> out;
    out.reserve(mesh.faces.size() * 24);
    for (const auto& f : mesh.faces) {
        glm::vec3 faceN = faceNormal(mesh.positions[f.x], mesh.positions[f.y], mesh.positions[f.z]);
        for (int idx : {f.x, f.y, f.z}) {
            const glm::vec3& p = mesh.positions[idx];
            glm::vec3 n = mesh.normals.empty() ? faceN : mesh.normals[idx];
            glm::vec2 uv = mesh.uvs.empty() ? glm::vec2(0.0f) : mesh.uvs[idx];
            out.push_back(p.x);
            out.push_back(p.y);
            out.push_back(p.z);
            out.push_back(n.x);
            out.push_back(n.y);
            out.push_back(n.z);
            out.push_back(uv.x);
            out.push_back(uv.y);
        }
    }
    return out;
}

GLuint makeTexture(const ImageData& image) {
    if (image.pixels.empty()) return 0;
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        image.width,
        image.height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        image.pixels.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

const char* VS_BODY = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
uniform mat4 MVP;
out vec3 vNorm;
out vec2 vUV;
void main() {
    vNorm = aNormal;
    vUV = aUV;
    gl_Position = MVP * vec4(aPos, 1.0);
}
)";

const char* FS_BODY = R"(
#version 330 core
in vec3 vNorm;
in vec2 vUV;
out vec4 FragColor;
uniform vec4 uBaseColorFactor;
uniform sampler2D uBaseColorTex;
uniform bool uUseTexture;
void main() {
    vec3 base = uBaseColorFactor.rgb;
    if (uUseTexture) {
        base *= texture(uBaseColorTex, vUV).rgb;
    }
    vec3 N = normalize(vNorm);
    vec3 L = normalize(vec3(1.0, 2.0, 1.5));
    float d = clamp(dot(N, L), 0.0, 1.0);
    vec3 col = mix(base * 0.55, base, d * 0.7 + 0.3);
    FragColor = vec4(col, uBaseColorFactor.a);
}
)";

const char* VS_FUR = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
layout(location=2) in float aAlong;
uniform mat4 MVP;
out vec3 vColor;
out float vAlong;
void main() {
    vColor = aColor;
    vAlong = aAlong;
    gl_Position = MVP * vec4(aPos, 1.0);
}
)";

const char* FS_FUR = R"(
#version 330 core
in vec3 vColor;
in float vAlong;
out vec4 FragColor;
void main() {
    vec3 root = vColor * 0.90;
    vec3 mid = vColor * 0.98;
    vec3 tip = mix(vColor, vec3(1.0), 0.04);
    vec3 col = (vAlong < 0.5) ? mix(root, mid, vAlong * 2.0)
                              : mix(mid, tip, (vAlong - 0.5) * 2.0);
    float alpha = mix(0.55, 0.02, pow(vAlong, 1.1));
    if (alpha < 0.025) discard;
    FragColor = vec4(col, alpha);
}
)";

int main(int argc, char** argv) {
    std::string glbPath = "assets/im_sol_arm_out.glb";
    if (argc > 1) glbPath = argv[1];

    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(W, H, "Low-Poly Cat with Fur", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD init failed\n";
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    SceneData scene = loadGLB(glbPath);
    if (scene.submeshes.empty()) return -1;

    glm::vec3 sceneCenter = 0.5f * (scene.boundsMin + scene.boundsMax);
    glm::vec3 sceneExtent = scene.boundsMax - scene.boundsMin;
    float sceneRadius = 0.5f * glm::length(sceneExtent);
    orbitTarget = sceneCenter + glm::vec3(0.0f, -sceneExtent.y * 0.02f, 0.0f);
    orbitDist = std::max(0.8f, sceneRadius / std::tan(glm::radians(22.5f)) * 1.45f);

    std::vector<GLuint> textures(scene.images.size(), 0);
    for (size_t i = 0; i < scene.images.size(); ++i) textures[i] = makeTexture(scene.images[i]);

    struct GPUMesh {
        GLuint vao = 0;
        GLuint vbo = 0;
        GLsizei count = 0;
        int materialIndex = 0;
        bool hasFur = false;
    };

    std::vector<GPUMesh> bodyGPU;
    std::vector<GPUMesh> furGPU;
    bodyGPU.reserve(scene.submeshes.size());
    furGPU.reserve(scene.submeshes.size());

    for (const auto& sub : scene.submeshes) {
        GPUMesh body;
        body.materialIndex = sub.materialIndex;
        auto bodyVerts = flattenMesh(sub);
        glGenVertexArrays(1, &body.vao);
        glGenBuffers(1, &body.vbo);
        glBindVertexArray(body.vao);
        glBindBuffer(GL_ARRAY_BUFFER, body.vbo);
        glBufferData(GL_ARRAY_BUFFER, bodyVerts.size() * sizeof(float), bodyVerts.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        body.count = static_cast<GLsizei>(bodyVerts.size() / 8);
        bodyGPU.push_back(body);

        GPUMesh fur;
        fur.materialIndex = sub.materialIndex;
        const MaterialData& mat = scene.materials[std::clamp(sub.materialIndex, 0, static_cast<int>(scene.materials.size()) - 1)];
        const ImageData* image = (mat.textureIndex >= 0 && mat.textureIndex < static_cast<int>(scene.images.size()))
                                     ? &scene.images[mat.textureIndex]
                                     : nullptr;
        bool isHeadMesh = (sub.name == "Roundcube");
        bool isBodyMesh = (sub.name == "Roundcube.001");
        bool shouldGenerateFur = isHeadMesh || isBodyMesh;
        auto furVerts = shouldGenerateFur ? generateFur(sub, mat, image, isHeadMesh) : std::vector<float>{};
        if (!furVerts.empty()) {
            glGenVertexArrays(1, &fur.vao);
            glGenBuffers(1, &fur.vbo);
            glBindVertexArray(fur.vao);
            glBindBuffer(GL_ARRAY_BUFFER, fur.vbo);
            glBufferData(GL_ARRAY_BUFFER, furVerts.size() * sizeof(float), furVerts.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
            glEnableVertexAttribArray(2);
            fur.count = static_cast<GLsizei>(furVerts.size() / 7);
            fur.hasFur = true;
        }
        furGPU.push_back(fur);
    }

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    GLuint bodyProg = makeProgram(VS_BODY, FS_BODY);
    GLuint furProg = makeProgram(VS_FUR, FS_FUR);

    GLint bodyMVP = glGetUniformLocation(bodyProg, "MVP");
    GLint baseColorFactorLoc = glGetUniformLocation(bodyProg, "uBaseColorFactor");
    GLint useTextureLoc = glGetUniformLocation(bodyProg, "uUseTexture");
    GLint baseTexLoc = glGetUniformLocation(bodyProg, "uBaseColorTex");
    GLint furMVP = glGetUniformLocation(furProg, "MVP");

    glUseProgram(bodyProg);
    glUniform1i(baseTexLoc, 0);

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        int fbW = W;
        int fbH = H;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        float aspect = fbH > 0 ? static_cast<float>(fbW) / static_cast<float>(fbH) : static_cast<float>(W) / static_cast<float>(H);

        glClearColor(0.08f, 0.08f, 0.13f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float yawR = glm::radians(orbitYaw);
        float pitchR = glm::radians(orbitPitch);
        glm::vec3 eye = orbitTarget + orbitDist * glm::vec3(
            std::cos(pitchR) * std::sin(yawR),
            std::sin(pitchR),
            std::cos(pitchR) * std::cos(yawR));
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 100.0f);
        glm::mat4 view = glm::lookAt(eye, orbitTarget, glm::vec3(0, 1, 0));
        glm::mat4 mvp = proj * view;

        glUseProgram(bodyProg);
        glUniformMatrix4fv(bodyMVP, 1, GL_FALSE, glm::value_ptr(mvp));
        for (const auto& mesh : bodyGPU) {
            const MaterialData& mat = scene.materials[std::clamp(mesh.materialIndex, 0, static_cast<int>(scene.materials.size()) - 1)];
            bool useTexture = mat.textureIndex >= 0 && mat.textureIndex < static_cast<int>(textures.size()) &&
                              textures[mat.textureIndex] != 0;
            glUniform4fv(baseColorFactorLoc, 1, glm::value_ptr(mat.baseColorFactor));
            glUniform1i(useTextureLoc, useTexture ? 1 : 0);
            if (useTexture) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, textures[mat.textureIndex]);
            } else {
                glBindTexture(GL_TEXTURE_2D, 0);
            }
            glBindVertexArray(mesh.vao);
            glDrawArrays(GL_TRIANGLES, 0, mesh.count);
        }

        glDepthMask(GL_FALSE);
        glUseProgram(furProg);
        glUniformMatrix4fv(furMVP, 1, GL_FALSE, glm::value_ptr(mvp));
        for (const auto& mesh : furGPU) {
            if (!mesh.hasFur || mesh.count == 0) continue;
            glBindVertexArray(mesh.vao);
            glDrawArrays(GL_LINES, 0, mesh.count);
        }
        glDepthMask(GL_TRUE);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    for (auto& mesh : bodyGPU) {
        glDeleteVertexArrays(1, &mesh.vao);
        glDeleteBuffers(1, &mesh.vbo);
    }
    for (auto& mesh : furGPU) {
        if (mesh.vao) glDeleteVertexArrays(1, &mesh.vao);
        if (mesh.vbo) glDeleteBuffers(1, &mesh.vbo);
    }
    for (GLuint tex : textures) {
        if (tex) glDeleteTextures(1, &tex);
    }

    glDeleteProgram(bodyProg);
    glDeleteProgram(furProg);
    glfwTerminate();
    return 0;
}
