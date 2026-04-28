#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <png.h>

#include "audio_player.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Mat4 {
    std::array<float, 16> m{};
};

struct Bounds {
    Vec3 min{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    Vec3 max{
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max()
    };
};

struct Particle {
    float x;
    float y;
    float z;
    float speed;
    float drift;
    float phase;
    float size;
};

struct ParticleVertex {
    float x;
    float y;
    float z;
    float size;
};

struct BokehParticle {
    float x;
    float y;
    float speed;
    float drift;
    float phase;
    float size;
    float alpha;
    float colorSeed;
    float twinkle;
    float flicker;
};

struct BokehVertex {
    float x;
    float y;
    float size;
    float alpha;
    float colorSeed;
    float twinkle;
    float flicker;
};

struct FlatVertex {
    float x;
    float y;
    float z;
    float r;
    float g;
    float b;
};

struct MeshVertex {
    float x;
    float y;
    float z;
    float nx;
    float ny;
    float nz;
    float r;
    float g;
    float b;
    float u;
    float v;
};

struct UmbrellaMesh {
    std::vector<MeshVertex> vertices;
    std::vector<Vec3> canopyTriangles;
    Bounds bounds;
};

struct SceneMesh {
    std::vector<MeshVertex> vertices;
    Bounds bounds;
};

struct TextureImage {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgba;
};

struct UmbrellaSilhouette {
    bool valid = false;
    float xMin = 0.0f;
    float xMax = 0.0f;
    std::array<float, 48> topY{};
};

struct UmbrellaHeightField {
    static constexpr std::size_t kSamples = 40;
    bool valid = false;
    float xMin = 0.0f;
    float xMax = 0.0f;
    float zMin = 0.0f;
    float zMax = 0.0f;
    std::array<float, kSamples * kSamples> topY{};
};

enum class CameraMode {
    Front = 0,
    Left,
    Right,
    Back,
    Above,
    Count
};

constexpr int kWindowWidth = 960;
constexpr int kWindowHeight = 640;
constexpr std::size_t kParticleCount = 900;
constexpr std::size_t kBokehCount = 56;
constexpr std::size_t kSnowSamples = 48;
constexpr float kFloorTopY = -0.54f;
constexpr float kFloorBottomY = -1.02f;

std::string trim(const std::string& value) {
    const std::size_t start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }

    const std::size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

std::string findFirstExistingPath(const std::vector<std::string>& candidates) {
    for (const std::string& path : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) {
            return path;
        }
    }

    return "";
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    (void)window;
    glViewport(0, 0, width, height);
}

Mat4 identityMatrix() {
    Mat4 matrix{};
    matrix.m[0] = 1.0f;
    matrix.m[5] = 1.0f;
    matrix.m[10] = 1.0f;
    matrix.m[15] = 1.0f;
    return matrix;
}

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 result{};
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            float value = 0.0f;
            for (int k = 0; k < 4; ++k) {
                value += a.m[k * 4 + row] * b.m[column * 4 + k];
            }
            result.m[column * 4 + row] = value;
        }
    }
    return result;
}

Mat4 translation(float x, float y, float z) {
    Mat4 matrix = identityMatrix();
    matrix.m[12] = x;
    matrix.m[13] = y;
    matrix.m[14] = z;
    return matrix;
}

Mat4 uniformScale(float scale) {
    Mat4 matrix = identityMatrix();
    matrix.m[0] = scale;
    matrix.m[5] = scale;
    matrix.m[10] = scale;
    return matrix;
}

Mat4 scale(float sx, float sy, float sz) {
    Mat4 matrix = identityMatrix();
    matrix.m[0] = sx;
    matrix.m[5] = sy;
    matrix.m[10] = sz;
    return matrix;
}

Mat4 rotationX(float angle) {
    Mat4 matrix = identityMatrix();
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    matrix.m[5] = c;
    matrix.m[9] = -s;
    matrix.m[6] = s;
    matrix.m[10] = c;
    return matrix;
}

Mat4 rotationY(float angle) {
    Mat4 matrix = identityMatrix();
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    matrix.m[0] = c;
    matrix.m[8] = s;
    matrix.m[2] = -s;
    matrix.m[10] = c;
    return matrix;
}

Mat4 rotationZ(float angle) {
    Mat4 matrix = identityMatrix();
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    matrix.m[0] = c;
    matrix.m[4] = -s;
    matrix.m[1] = s;
    matrix.m[5] = c;
    return matrix;
}

Vec3 transformPoint(const Mat4& matrix, const Vec3& point) {
    return {
        matrix.m[0] * point.x + matrix.m[4] * point.y + matrix.m[8] * point.z + matrix.m[12],
        matrix.m[1] * point.x + matrix.m[5] * point.y + matrix.m[9] * point.z + matrix.m[13],
        matrix.m[2] * point.x + matrix.m[6] * point.y + matrix.m[10] * point.z + matrix.m[14]
    };
}

void expandBounds(Bounds& bounds, const Vec3& point) {
    bounds.min.x = std::min(bounds.min.x, point.x);
    bounds.min.y = std::min(bounds.min.y, point.y);
    bounds.min.z = std::min(bounds.min.z, point.z);
    bounds.max.x = std::max(bounds.max.x, point.x);
    bounds.max.y = std::max(bounds.max.y, point.y);
    bounds.max.z = std::max(bounds.max.z, point.z);
}

bool isCanopyMaterial(const std::string& materialName) {
    return materialName == "umbrella_3" || materialName == "umbrella_4";
}

Vec3 umbrellaColorForVertex(const std::string& materialName, const Vec3& point) {
    // The imported mesh mixes canopy and shaft pieces across materials.
    // Vertices lower on the model belong to the handle/stick and should stay dark.
    if (point.z < 18.0f) {
        return {0.18f, 0.12f, 0.08f};
    }

    if (isCanopyMaterial(materialName)) {
        return {0.92f, 0.78f, 0.24f};
    }
    if (materialName == "umbrella_1") {
        return {0.22f, 0.16f, 0.10f};
    }
    if (materialName == "umbrella_2") {
        return {0.16f, 0.10f, 0.06f};
    }
    return {0.30f, 0.20f, 0.12f};
}

bool parseFaceToken(const std::string& token, int& positionIndex, int& normalIndex) {
    positionIndex = 0;
    normalIndex = 0;

    const std::size_t firstSlash = token.find('/');
    if (firstSlash == std::string::npos) {
        positionIndex = std::stoi(token);
        return true;
    }

    positionIndex = std::stoi(token.substr(0, firstSlash));
    const std::size_t secondSlash = token.find('/', firstSlash + 1);
    if (secondSlash == std::string::npos) {
        return true;
    }

    if (secondSlash + 1 < token.size()) {
        normalIndex = std::stoi(token.substr(secondSlash + 1));
    }
    return true;
}

bool loadUmbrellaMesh(const std::string& path, UmbrellaMesh& mesh) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    positions.reserve(160000);
    normals.reserve(90000);
    mesh.vertices.clear();
    mesh.canopyTriangles.clear();
    mesh.bounds = Bounds{};

    std::string currentMaterial = "umbrella_1";
    std::string line;
    while (std::getline(input, line)) {
        if (line.size() < 2) {
            continue;
        }

        if (line.rfind("v ", 0) == 0) {
            std::istringstream stream(line.substr(2));
            Vec3 point{};
            stream >> point.x >> point.y >> point.z;
            positions.push_back(point);
            expandBounds(mesh.bounds, point);
            continue;
        }

        if (line.rfind("vn ", 0) == 0) {
            std::istringstream stream(line.substr(3));
            Vec3 normal{};
            stream >> normal.x >> normal.y >> normal.z;
            normals.push_back(normal);
            continue;
        }

        if (line.rfind("usemtl ", 0) == 0) {
            currentMaterial = trim(line.substr(7));
            continue;
        }

        if (line.rfind("f ", 0) != 0) {
            continue;
        }

        std::istringstream stream(line.substr(2));
        std::vector<std::pair<int, int>> face;
        std::string token;
        while (stream >> token) {
            int positionIndex = 0;
            int normalIndex = 0;
            if (parseFaceToken(token, positionIndex, normalIndex)) {
                face.emplace_back(positionIndex - 1, normalIndex - 1);
            }
        }

        if (face.size() < 3) {
            continue;
        }

        for (std::size_t i = 1; i + 1 < face.size(); ++i) {
            const std::array<std::pair<int, int>, 3> tri = {face[0], face[i], face[i + 1]};
            for (const auto& [posIndex, normalIndex] : tri) {
                if (posIndex < 0 || static_cast<std::size_t>(posIndex) >= positions.size()) {
                    continue;
                }

                const Vec3& point = positions[static_cast<std::size_t>(posIndex)];
                const Vec3 color = umbrellaColorForVertex(currentMaterial, point);
                Vec3 normal{0.0f, 0.0f, 1.0f};
                if (normalIndex >= 0 && static_cast<std::size_t>(normalIndex) < normals.size()) {
                    normal = normals[static_cast<std::size_t>(normalIndex)];
                }

                mesh.vertices.push_back({
                    point.x, point.y, point.z,
                    normal.x, normal.y, normal.z,
                    color.x, color.y, color.z,
                    0.0f, 0.0f
                });

                if (isCanopyMaterial(currentMaterial)) {
                    mesh.canopyTriangles.push_back(point);
                }
            }
        }
    }

    return !mesh.vertices.empty();
}

bool decodePngFromMemory(const unsigned char* data, std::size_t size, TextureImage& texture) {
    png_image image{};
    image.version = PNG_IMAGE_VERSION;

    if (!png_image_begin_read_from_memory(&image, data, size)) {
        return false;
    }

    image.format = PNG_FORMAT_RGBA;
    texture.width = static_cast<int>(image.width);
    texture.height = static_cast<int>(image.height);
    texture.rgba.resize(PNG_IMAGE_SIZE(image));

    if (!png_image_finish_read(&image, nullptr, texture.rgba.data(), 0, nullptr)) {
        png_image_free(&image);
        texture = TextureImage{};
        return false;
    }

    png_image_free(&image);
    return true;
}

GLuint createTexture2D(const TextureImage& image) {
    if (image.width <= 0 || image.height <= 0 || image.rgba.empty()) {
        return 0;
    }

    GLuint textureId = 0;
    glGenTextures(1, &textureId);
    glBindTexture(GL_TEXTURE_2D, textureId);
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
        image.rgba.data()
    );
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    return textureId;
}

aiVector3D transformPointByMatrix(const aiMatrix4x4& matrix, const aiVector3D& point) {
    return {
        matrix.a1 * point.x + matrix.a2 * point.y + matrix.a3 * point.z + matrix.a4,
        matrix.b1 * point.x + matrix.b2 * point.y + matrix.b3 * point.z + matrix.b4,
        matrix.c1 * point.x + matrix.c2 * point.y + matrix.c3 * point.z + matrix.c4
    };
}

aiVector3D transformDirectionByMatrix(const aiMatrix4x4& matrix, const aiVector3D& direction) {
    return {
        matrix.a1 * direction.x + matrix.a2 * direction.y + matrix.a3 * direction.z,
        matrix.b1 * direction.x + matrix.b2 * direction.y + matrix.b3 * direction.z,
        matrix.c1 * direction.x + matrix.c2 * direction.y + matrix.c3 * direction.z
    };
}

void collectNodeTransforms(
    const aiNode* node,
    const aiMatrix4x4& parentTransform,
    std::unordered_map<std::string, aiMatrix4x4>& nodeTransforms,
    std::vector<aiMatrix4x4>& meshTransforms
) {
    if (!node) {
        return;
    }

    const aiMatrix4x4 globalTransform = parentTransform * node->mTransformation;
    nodeTransforms[node->mName.C_Str()] = globalTransform;

    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        const unsigned int meshIndex = node->mMeshes[i];
        if (meshIndex < meshTransforms.size()) {
            meshTransforms[meshIndex] = globalTransform;
        }
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        collectNodeTransforms(node->mChildren[i], globalTransform, nodeTransforms, meshTransforms);
    }
}

GLuint createSolidWhiteTexture() {
    TextureImage white{};
    white.width = 1;
    white.height = 1;
    white.rgba = {255, 255, 255, 255};
    return createTexture2D(white);
}

Mat4 buildUmbrellaModelMatrix(const Bounds& bounds) {
    const Vec3 center{
        (bounds.min.x + bounds.max.x) * 0.5f,
        (bounds.min.y + bounds.max.y) * 0.5f,
        (bounds.min.z + bounds.max.z) * 0.5f
    };

    Mat4 oriented = identityMatrix();
    oriented = multiply(rotationZ(-0.2f), oriented);
    oriented = multiply(rotationY(-0.1f), oriented);
    oriented = multiply(rotationX(-1.55f), oriented);
    oriented = multiply(translation(-center.x, -center.y, -center.z), oriented);

    const std::array<Vec3, 8> corners = {
        Vec3{bounds.min.x, bounds.min.y, bounds.min.z},
        Vec3{bounds.min.x, bounds.min.y, bounds.max.z},
        Vec3{bounds.min.x, bounds.max.y, bounds.min.z},
        Vec3{bounds.min.x, bounds.max.y, bounds.max.z},
        Vec3{bounds.max.x, bounds.min.y, bounds.min.z},
        Vec3{bounds.max.x, bounds.min.y, bounds.max.z},
        Vec3{bounds.max.x, bounds.max.y, bounds.min.z},
        Vec3{bounds.max.x, bounds.max.y, bounds.max.z}
    };

    Bounds orientedBounds{};
    for (const Vec3& corner : corners) {
        expandBounds(orientedBounds, transformPoint(oriented, corner));
    }

    const float orientedExtentX = orientedBounds.max.x - orientedBounds.min.x;
    const float orientedExtentY = orientedBounds.max.y - orientedBounds.min.y;
    const float orientedExtentZ = orientedBounds.max.z - orientedBounds.min.z;
    const float orientedMaxExtent = std::max({orientedExtentX, orientedExtentY, orientedExtentZ});
    const float targetSize = 1.00f;
    const float fittedScale = orientedMaxExtent > 0.0001f ? (targetSize / orientedMaxExtent) : 1.0f;

    Mat4 fitted = multiply(uniformScale(fittedScale), oriented);

    Bounds fittedBounds{};
    for (const Vec3& corner : corners) {
        expandBounds(fittedBounds, transformPoint(fitted, corner));
    }

    const Vec3 fittedCenter{
        (fittedBounds.min.x + fittedBounds.max.x) * 0.5f,
        (fittedBounds.min.y + fittedBounds.max.y) * 0.5f,
        (fittedBounds.min.z + fittedBounds.max.z) * 0.5f
    };

    const float targetCenterX = 3.00f;
    const float targetBottomY = -25.00f;
    const float targetCenterZ = -17.0f;

    Mat4 model = identityMatrix();
    model = multiply(
        translation(
            targetCenterX - fittedCenter.x,
            targetBottomY - fittedBounds.min.y,
            targetCenterZ - fittedCenter.z
        ),
        model
    );
    model = multiply(fitted, model);
    return model;
}

Mat4 buildCharacterModelMatrix(const Bounds& bounds) {
    const Vec3 center{
        (bounds.min.x + bounds.max.x) * 0.5f,
        (bounds.min.y + bounds.max.y) * 0.5f,
        (bounds.min.z + bounds.max.z) * 0.5f
    };

    const float extentX = bounds.max.x - bounds.min.x;
    const float extentY = bounds.max.y - bounds.min.y;
    const float extentZ = bounds.max.z - bounds.min.z;
    const float maxExtent = std::max({extentX, extentY, extentZ});
    const float normalizedScale = maxExtent > 0.0001f ? (0.62f / maxExtent) : 1.0f;
    const float halfHeightScaled = extentY * 0.5f * normalizedScale;
    const float characterY = (kFloorTopY + 0.01f) + halfHeightScaled;

    Mat4 model = identityMatrix();
    model = multiply(rotationY(-1.57f), model);
    model = multiply(scale(1.0f, 1.0f, -1.0f), model);
    model = multiply(uniformScale(normalizedScale), model);
    model = multiply(translation(-center.x, -center.y, -center.z), model);
    // Apply final placement in world space so X/Y/Z edits move the model predictably on screen.
    model = multiply(translation(0.12f, characterY, -0.04f), model);
    return model;
}

Mat4 buildSunjaeModelMatrix(const Bounds& bounds) {
    const Vec3 center{
        (bounds.min.x + bounds.max.x) * 0.5f,
        (bounds.min.y + bounds.max.y) * 0.5f,
        (bounds.min.z + bounds.max.z) * 0.5f
    };

    const float extentX = bounds.max.x - bounds.min.x;
    const float extentY = bounds.max.y - bounds.min.y;
    const float extentZ = bounds.max.z - bounds.min.z;
    const float maxExtent = std::max({extentX, extentY, extentZ});
    const float normalizedScale = maxExtent > 0.0001f ? (0.8f / maxExtent) : 1.0f;
    const float halfHeightScaled = extentY * 0.5f * normalizedScale;
    const float characterY = (kFloorTopY + 0.01f) + halfHeightScaled;

    Mat4 model = identityMatrix();
    model = multiply(rotationY(1.30f), model);
    model = multiply(scale(1.0f, 1.0f, -1.0f), model);
    model = multiply(uniformScale(normalizedScale), model);
    model = multiply(translation(-center.x, -center.y, -center.z), model);
    model = multiply(translation(-0.18f, characterY - 0.18f, -0.10f), model);
    return model;
}

Mat4 buildCameraViewTransform(CameraMode mode) {
    switch (mode) {
        case CameraMode::Front:
            return identityMatrix();
        case CameraMode::Left:
            return rotationY(1.57f);
        case CameraMode::Right:
            return rotationY(-1.57f);
        case CameraMode::Back:
            return rotationY(3.14159f);
        case CameraMode::Above:
            return multiply(rotationX(-1.20f), translation(0.0f, -0.10f, 0.0f));
        case CameraMode::Count:
            break;
    }

    return identityMatrix();
}

bool loadGlbMesh(const std::string& path, SceneMesh& mesh, TextureImage& embeddedTexture) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
        path,
        aiProcess_Triangulate |
            aiProcess_GenNormals |
            aiProcess_ImproveCacheLocality
    );

    if (!scene || !scene->HasMeshes()) {
        return false;
    }

    mesh.vertices.clear();
    mesh.bounds = Bounds{};
    embeddedTexture = TextureImage{};

    std::unordered_map<std::string, aiMatrix4x4> nodeTransforms;
    std::vector<aiMatrix4x4> meshTransforms(scene->mNumMeshes, aiMatrix4x4());
    collectNodeTransforms(scene->mRootNode, aiMatrix4x4(), nodeTransforms, meshTransforms);

    if (scene->HasMaterials()) {
        for (unsigned int materialIndex = 0; materialIndex < scene->mNumMaterials; ++materialIndex) {
            const aiMaterial* material = scene->mMaterials[materialIndex];
            aiString textureRef;
            aiReturn hasTexture = material->GetTexture(aiTextureType_BASE_COLOR, 0, &textureRef);
            if (hasTexture != AI_SUCCESS) {
                hasTexture = material->GetTexture(aiTextureType_DIFFUSE, 0, &textureRef);
            }

            if (hasTexture != AI_SUCCESS) {
                continue;
            }

            const std::string textureName = textureRef.C_Str();
            if (!textureName.empty() && textureName[0] == '*') {
                const long index = std::strtol(textureName.c_str() + 1, nullptr, 10);
                if (index < 0 || static_cast<unsigned int>(index) >= scene->mNumTextures) {
                    continue;
                }

                const aiTexture* texture = scene->mTextures[static_cast<unsigned int>(index)];
                if (!texture) {
                    continue;
                }

                if (texture->mHeight == 0) {
                    const auto* bytes = reinterpret_cast<const unsigned char*>(texture->pcData);
                    if (decodePngFromMemory(bytes, texture->mWidth, embeddedTexture)) {
                        break;
                    }
                } else {
                    embeddedTexture.width = static_cast<int>(texture->mWidth);
                    embeddedTexture.height = static_cast<int>(texture->mHeight);
                    embeddedTexture.rgba.resize(static_cast<std::size_t>(texture->mWidth) * static_cast<std::size_t>(texture->mHeight) * 4);
                    for (unsigned int pixel = 0; pixel < texture->mWidth * texture->mHeight; ++pixel) {
                        const aiTexel& texel = texture->pcData[pixel];
                        const std::size_t base = static_cast<std::size_t>(pixel) * 4;
                        embeddedTexture.rgba[base + 0] = texel.r;
                        embeddedTexture.rgba[base + 1] = texel.g;
                        embeddedTexture.rgba[base + 2] = texel.b;
                        embeddedTexture.rgba[base + 3] = texel.a;
                    }
                    break;
                }
            }
        }
    }

    for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex) {
        const aiMesh* aiMeshData = scene->mMeshes[meshIndex];
        if (!aiMeshData || !aiMeshData->HasPositions()) {
            continue;
        }

        const aiMatrix4x4 meshTransform = meshTransforms[meshIndex];

        Vec3 materialColor{1.0f, 1.0f, 1.0f};
        if (scene->HasMaterials() && aiMeshData->mMaterialIndex < scene->mNumMaterials) {
            const aiMaterial* material = scene->mMaterials[aiMeshData->mMaterialIndex];
            aiColor4D color{};
            if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_BASE_COLOR, &color) ||
                AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &color)) {
                materialColor = {color.r, color.g, color.b};
            }
        }

        std::vector<aiVector3D> deformedPositions(aiMeshData->mNumVertices);
        std::vector<aiVector3D> deformedNormals(aiMeshData->mNumVertices);

        if (aiMeshData->HasBones()) {
            std::vector<aiVector3D> skinnedPositions(aiMeshData->mNumVertices, aiVector3D(0.0f, 0.0f, 0.0f));
            std::vector<aiVector3D> skinnedNormals(aiMeshData->mNumVertices, aiVector3D(0.0f, 0.0f, 0.0f));
            std::vector<float> totalWeights(aiMeshData->mNumVertices, 0.0f);

            for (unsigned int boneIndex = 0; boneIndex < aiMeshData->mNumBones; ++boneIndex) {
                const aiBone* bone = aiMeshData->mBones[boneIndex];
                if (!bone) {
                    continue;
                }

                const auto foundBone = nodeTransforms.find(bone->mName.C_Str());
                if (foundBone == nodeTransforms.end()) {
                    continue;
                }

                const aiMatrix4x4 skinTransform = foundBone->second * bone->mOffsetMatrix;

                for (unsigned int weightIndex = 0; weightIndex < bone->mNumWeights; ++weightIndex) {
                    const aiVertexWeight& weight = bone->mWeights[weightIndex];
                    if (weight.mVertexId >= aiMeshData->mNumVertices || weight.mWeight <= 0.0f) {
                        continue;
                    }

                    const unsigned int vertexIndex = weight.mVertexId;
                    const aiVector3D basePosition = aiMeshData->mVertices[vertexIndex];
                    const aiVector3D baseNormal = aiMeshData->HasNormals()
                        ? aiMeshData->mNormals[vertexIndex]
                        : aiVector3D(0.0f, 0.0f, 1.0f);

                    skinnedPositions[vertexIndex] += transformPointByMatrix(skinTransform, basePosition) * weight.mWeight;
                    skinnedNormals[vertexIndex] += transformDirectionByMatrix(skinTransform, baseNormal) * weight.mWeight;
                    totalWeights[vertexIndex] += weight.mWeight;
                }
            }

            for (unsigned int vertexIndex = 0; vertexIndex < aiMeshData->mNumVertices; ++vertexIndex) {
                const aiVector3D basePosition = aiMeshData->mVertices[vertexIndex];
                const aiVector3D baseNormal = aiMeshData->HasNormals()
                    ? aiMeshData->mNormals[vertexIndex]
                    : aiVector3D(0.0f, 0.0f, 1.0f);
                const float weight = totalWeights[vertexIndex];

                if (weight > 0.0f) {
                    const float residual = std::max(0.0f, 1.0f - weight);
                    deformedPositions[vertexIndex] = skinnedPositions[vertexIndex] + transformPointByMatrix(meshTransform, basePosition) * residual;
                    deformedNormals[vertexIndex] = skinnedNormals[vertexIndex] + transformDirectionByMatrix(meshTransform, baseNormal) * residual;
                } else {
                    deformedPositions[vertexIndex] = transformPointByMatrix(meshTransform, basePosition);
                    deformedNormals[vertexIndex] = transformDirectionByMatrix(meshTransform, baseNormal);
                }

                if (deformedNormals[vertexIndex].SquareLength() > 0.0f) {
                    deformedNormals[vertexIndex].Normalize();
                } else {
                    deformedNormals[vertexIndex] = aiVector3D(0.0f, 0.0f, 1.0f);
                }
            }
        } else {
            for (unsigned int vertexIndex = 0; vertexIndex < aiMeshData->mNumVertices; ++vertexIndex) {
                const aiVector3D basePosition = aiMeshData->mVertices[vertexIndex];
                const aiVector3D baseNormal = aiMeshData->HasNormals()
                    ? aiMeshData->mNormals[vertexIndex]
                    : aiVector3D(0.0f, 0.0f, 1.0f);
                deformedPositions[vertexIndex] = transformPointByMatrix(meshTransform, basePosition);
                deformedNormals[vertexIndex] = transformDirectionByMatrix(meshTransform, baseNormal);
                if (deformedNormals[vertexIndex].SquareLength() > 0.0f) {
                    deformedNormals[vertexIndex].Normalize();
                } else {
                    deformedNormals[vertexIndex] = aiVector3D(0.0f, 0.0f, 1.0f);
                }
            }
        }

        for (unsigned int faceIndex = 0; faceIndex < aiMeshData->mNumFaces; ++faceIndex) {
            const aiFace& face = aiMeshData->mFaces[faceIndex];
            if (face.mNumIndices != 3) {
                continue;
            }

            for (unsigned int localIndex = 0; localIndex < 3; ++localIndex) {
                const unsigned int index = face.mIndices[localIndex];
                if (index >= aiMeshData->mNumVertices) {
                    continue;
                }

                const aiVector3D& position = deformedPositions[index];
                const aiVector3D& aiNormal = deformedNormals[index];
                const Vec3 normal{aiNormal.x, aiNormal.y, aiNormal.z};

                Vec3 color = materialColor;
                if (aiMeshData->HasVertexColors(0)) {
                    const aiColor4D& vertexColor = aiMeshData->mColors[0][index];
                    color = {vertexColor.r, vertexColor.g, vertexColor.b};
                }

                float u = 0.0f;
                float v = 0.0f;
                if (aiMeshData->HasTextureCoords(0)) {
                    const aiVector3D& texCoord = aiMeshData->mTextureCoords[0][index];
                    u = texCoord.x;
                    v = 1.0f - texCoord.y;
                }

                mesh.vertices.push_back({
                    position.x, position.y, position.z,
                    normal.x, normal.y, normal.z,
                    color.x, color.y, color.z,
                    u, v
                });
                expandBounds(mesh.bounds, {position.x, position.y, position.z});
            }
        }
    }

    return !mesh.vertices.empty();
}

UmbrellaSilhouette buildSilhouette(const std::vector<Vec3>& canopyTriangles, const Mat4& modelMatrix) {
    UmbrellaSilhouette silhouette{};
    silhouette.topY.fill(-10.0f);

    if (canopyTriangles.empty()) {
        return silhouette;
    }

    std::vector<Vec3> transformed;
    transformed.reserve(canopyTriangles.size());
    silhouette.xMin = std::numeric_limits<float>::max();
    silhouette.xMax = -std::numeric_limits<float>::max();

    for (const Vec3& point : canopyTriangles) {
        const Vec3 transformedPoint = transformPoint(modelMatrix, point);
        transformed.push_back(transformedPoint);
        silhouette.xMin = std::min(silhouette.xMin, transformedPoint.x);
        silhouette.xMax = std::max(silhouette.xMax, transformedPoint.x);
    }

    if (!(silhouette.xMax > silhouette.xMin)) {
        return silhouette;
    }

    for (std::size_t sample = 0; sample < kSnowSamples; ++sample) {
        const float t = static_cast<float>(sample) / static_cast<float>(kSnowSamples - 1);
        const float sampleX = silhouette.xMin + (silhouette.xMax - silhouette.xMin) * t;
        float bestY = -10.0f;

        for (std::size_t i = 0; i + 2 < transformed.size(); i += 3) {
            const std::array<Vec3, 3> tri = {transformed[i], transformed[i + 1], transformed[i + 2]};
            for (int edge = 0; edge < 3; ++edge) {
                const Vec3& a = tri[edge];
                const Vec3& b = tri[(edge + 1) % 3];
                const float minX = std::min(a.x, b.x);
                const float maxX = std::max(a.x, b.x);
                if (sampleX < minX || sampleX > maxX) {
                    continue;
                }

                const float dx = b.x - a.x;
                if (std::abs(dx) < 0.0001f) {
                    bestY = std::max(bestY, std::max(a.y, b.y));
                    continue;
                }

                const float alpha = (sampleX - a.x) / dx;
                if (alpha < 0.0f || alpha > 1.0f) {
                    continue;
                }

                bestY = std::max(bestY, a.y + (b.y - a.y) * alpha);
            }
        }

        silhouette.topY[sample] = bestY;
    }

    for (std::size_t i = 1; i + 1 < silhouette.topY.size(); ++i) {
        if (silhouette.topY[i] < -5.0f) {
            silhouette.topY[i] = std::max(silhouette.topY[i - 1], silhouette.topY[i + 1]);
        }
    }

    silhouette.valid = true;
    return silhouette;
}

bool triangleHeightAtXZ(
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    float x,
    float z,
    float& outY
) {
    const float x1 = a.x;
    const float z1 = a.z;
    const float x2 = b.x;
    const float z2 = b.z;
    const float x3 = c.x;
    const float z3 = c.z;

    const float denom = (z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3);
    if (std::abs(denom) < 0.000001f) {
        return false;
    }

    const float w1 = ((z2 - z3) * (x - x3) + (x3 - x2) * (z - z3)) / denom;
    const float w2 = ((z3 - z1) * (x - x3) + (x1 - x3) * (z - z3)) / denom;
    const float w3 = 1.0f - w1 - w2;

    if (w1 < -0.001f || w2 < -0.001f || w3 < -0.001f) {
        return false;
    }

    outY = w1 * a.y + w2 * b.y + w3 * c.y;
    return true;
}

UmbrellaHeightField buildUmbrellaHeightField(const std::vector<Vec3>& canopyTriangles, const Mat4& modelMatrix) {
    UmbrellaHeightField field{};
    field.topY.fill(-10.0f);

    if (canopyTriangles.empty()) {
        return field;
    }

    std::vector<Vec3> transformed;
    transformed.reserve(canopyTriangles.size());
    field.xMin = std::numeric_limits<float>::max();
    field.xMax = -std::numeric_limits<float>::max();
    field.zMin = std::numeric_limits<float>::max();
    field.zMax = -std::numeric_limits<float>::max();

    for (const Vec3& point : canopyTriangles) {
        const Vec3 transformedPoint = transformPoint(modelMatrix, point);
        transformed.push_back(transformedPoint);
        field.xMin = std::min(field.xMin, transformedPoint.x);
        field.xMax = std::max(field.xMax, transformedPoint.x);
        field.zMin = std::min(field.zMin, transformedPoint.z);
        field.zMax = std::max(field.zMax, transformedPoint.z);
    }

    if (!(field.xMax > field.xMin) || !(field.zMax > field.zMin)) {
        return field;
    }

    for (std::size_t zi = 0; zi < UmbrellaHeightField::kSamples; ++zi) {
        const float tz = static_cast<float>(zi) / static_cast<float>(UmbrellaHeightField::kSamples - 1);
        const float sampleZ = field.zMin + (field.zMax - field.zMin) * tz;
        for (std::size_t xi = 0; xi < UmbrellaHeightField::kSamples; ++xi) {
            const float tx = static_cast<float>(xi) / static_cast<float>(UmbrellaHeightField::kSamples - 1);
            const float sampleX = field.xMin + (field.xMax - field.xMin) * tx;

            float bestY = -10.0f;
            for (std::size_t i = 0; i + 2 < transformed.size(); i += 3) {
                float y = 0.0f;
                if (triangleHeightAtXZ(transformed[i], transformed[i + 1], transformed[i + 2], sampleX, sampleZ, y)) {
                    bestY = std::max(bestY, y);
                }
            }

            field.topY[zi * UmbrellaHeightField::kSamples + xi] = bestY;
        }
    }

    field.valid = true;
    return field;
}

float umbrellaTopY3D(const UmbrellaHeightField& field, float x, float z) {
    if (!field.valid || x < field.xMin || x > field.xMax || z < field.zMin || z > field.zMax) {
        return -10.0f;
    }

    const float nx = (x - field.xMin) / (field.xMax - field.xMin);
    const float nz = (z - field.zMin) / (field.zMax - field.zMin);
    const float fx = nx * static_cast<float>(UmbrellaHeightField::kSamples - 1);
    const float fz = nz * static_cast<float>(UmbrellaHeightField::kSamples - 1);

    const std::size_t x0 = static_cast<std::size_t>(fx);
    const std::size_t z0 = static_cast<std::size_t>(fz);
    const std::size_t x1 = std::min(x0 + 1, UmbrellaHeightField::kSamples - 1);
    const std::size_t z1 = std::min(z0 + 1, UmbrellaHeightField::kSamples - 1);

    const float ax = fx - static_cast<float>(x0);
    const float az = fz - static_cast<float>(z0);

    const float y00 = field.topY[z0 * UmbrellaHeightField::kSamples + x0];
    const float y10 = field.topY[z0 * UmbrellaHeightField::kSamples + x1];
    const float y01 = field.topY[z1 * UmbrellaHeightField::kSamples + x0];
    const float y11 = field.topY[z1 * UmbrellaHeightField::kSamples + x1];

    const float y0 = y00 + (y10 - y00) * ax;
    const float y1 = y01 + (y11 - y01) * ax;
    return y0 + (y1 - y0) * az;
}

bool hitsUmbrella3D(const UmbrellaHeightField& field, float x, float y, float z) {
    const float topY = umbrellaTopY3D(field, x, z);
    return topY > -5.0f && y <= topY + 0.010f && y >= topY - 0.06f;
}

Particle makeParticle(std::mt19937& rng, bool spawnAtTop) {
    std::uniform_real_distribution<float> xDist(-1.05f, 1.05f);
    std::uniform_real_distribution<float> yDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> zDist(-0.95f, 0.95f);
    std::uniform_real_distribution<float> speedDist(0.22f, 0.62f);
    std::uniform_real_distribution<float> driftDist(0.5f, 1.8f);
    std::uniform_real_distribution<float> phaseDist(0.0f, 6.28318f);
    std::uniform_real_distribution<float> sizeDist(4.0f, 9.0f);
    std::uniform_real_distribution<float> topOffsetDist(0.0f, 0.35f);

    Particle particle{};
    particle.x = xDist(rng);
    particle.y = spawnAtTop ? 1.05f + topOffsetDist(rng) : yDist(rng);
    particle.z = zDist(rng);
    particle.speed = speedDist(rng);
    particle.drift = driftDist(rng);
    particle.phase = phaseDist(rng);
    particle.size = sizeDist(rng);
    return particle;
}

BokehParticle makeBokehParticle(std::mt19937& rng, bool spawnAtTop) {
    std::uniform_real_distribution<float> xDist(-1.10f, 1.10f);
    std::uniform_real_distribution<float> yDist(-1.15f, 1.15f);
    std::uniform_real_distribution<float> speedDist(0.03f, 0.12f);
    std::uniform_real_distribution<float> driftDist(0.6f, 2.0f);
    std::uniform_real_distribution<float> phaseDist(0.0f, 6.28318f);
    std::uniform_real_distribution<float> sizeDist(96.0f, 260.0f);
    std::uniform_real_distribution<float> alphaDist(0.07f, 0.18f);
    std::uniform_real_distribution<float> colorSeedDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> twinkleDist(0.0f, 6.28318f);
    std::uniform_real_distribution<float> flickerDist(0.0f, 0.9f);
    std::bernoulli_distribution flickerChance(0.42);

    BokehParticle particle{};
    particle.x = xDist(rng);
    particle.y = spawnAtTop ? 1.20f : yDist(rng);
    particle.speed = speedDist(rng);
    particle.drift = driftDist(rng);
    particle.phase = phaseDist(rng);
    particle.size = sizeDist(rng);
    particle.alpha = alphaDist(rng);
    particle.colorSeed = colorSeedDist(rng);
    particle.twinkle = twinkleDist(rng);
    particle.flicker = flickerChance(rng) ? flickerDist(rng) : 0.0f;
    return particle;
}

std::vector<FlatVertex> buildBackground() {
    return {
        {-1.0f, -1.0f, 0.0f, 0.08f, 0.11f, 0.22f},
        { 1.0f, -1.0f, 0.0f, 0.08f, 0.11f, 0.22f},
        { 1.0f,  1.0f, 0.0f, 0.03f, 0.05f, 0.14f},
        {-1.0f, -1.0f, 0.0f, 0.08f, 0.11f, 0.22f},
        { 1.0f,  1.0f, 0.0f, 0.03f, 0.05f, 0.14f},
        {-1.0f,  1.0f, 0.0f, 0.03f, 0.05f, 0.14f},
    };
}

std::vector<FlatVertex> buildGroundBand() {
    return {
        {-1.0f, -1.0f, 0.0f, 0.80f, 0.85f, 0.93f},
        { 1.0f, -1.0f, 0.0f, 0.80f, 0.85f, 0.93f},
        { 1.0f, -0.42f, 0.0f, 0.78f, 0.83f, 0.92f},
        {-1.0f, -1.0f, 0.0f, 0.80f, 0.85f, 0.93f},
        { 1.0f, -0.42f, 0.0f, 0.78f, 0.83f, 0.92f},
        {-1.0f, -0.42f, 0.0f, 0.78f, 0.83f, 0.92f},
    };
}

std::vector<FlatVertex> buildTopViewFloorFill() {
    return {
        {-1.0f, -1.0f, 0.0f, 0.80f, 0.85f, 0.93f},
        { 1.0f, -1.0f, 0.0f, 0.80f, 0.85f, 0.93f},
        { 1.0f,  1.0f, 0.0f, 0.80f, 0.85f, 0.93f},
        {-1.0f, -1.0f, 0.0f, 0.80f, 0.85f, 0.93f},
        { 1.0f,  1.0f, 0.0f, 0.80f, 0.85f, 0.93f},
        {-1.0f,  1.0f, 0.0f, 0.80f, 0.85f, 0.93f},
    };
}

std::vector<MeshVertex> buildFloorMesh() {
    const Vec3 floorColor{0.84f, 0.88f, 0.96f};
    constexpr float xMin = -1.10f;
    constexpr float xMax = 1.10f;
    constexpr float zMin = -1.05f;
    constexpr float zMax = 1.05f;

    return {
        // Top face
        {xMin, kFloorTopY, zMin, 0.0f, 1.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 0.0f},
        {xMax, kFloorTopY, zMin, 0.0f, 1.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 0.0f},
        {xMax, kFloorTopY, zMax, 0.0f, 1.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 1.0f},
        {xMin, kFloorTopY, zMin, 0.0f, 1.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 0.0f},
        {xMax, kFloorTopY, zMax, 0.0f, 1.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 1.0f},
        {xMin, kFloorTopY, zMax, 0.0f, 1.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 1.0f},

        // Front side (z = zMax)
        {xMin, kFloorBottomY, zMax, 0.0f, 0.0f, 1.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 0.0f},
        {xMax, kFloorBottomY, zMax, 0.0f, 0.0f, 1.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 0.0f},
        {xMax, kFloorTopY, zMax, 0.0f, 0.0f, 1.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 1.0f},
        {xMin, kFloorBottomY, zMax, 0.0f, 0.0f, 1.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 0.0f},
        {xMax, kFloorTopY, zMax, 0.0f, 0.0f, 1.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 1.0f},
        {xMin, kFloorTopY, zMax, 0.0f, 0.0f, 1.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 1.0f},

        // Back side (z = zMin)
        {xMax, kFloorBottomY, zMin, 0.0f, 0.0f, -1.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 0.0f},
        {xMin, kFloorBottomY, zMin, 0.0f, 0.0f, -1.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 0.0f},
        {xMin, kFloorTopY, zMin, 0.0f, 0.0f, -1.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 1.0f},
        {xMax, kFloorBottomY, zMin, 0.0f, 0.0f, -1.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 0.0f},
        {xMin, kFloorTopY, zMin, 0.0f, 0.0f, -1.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 1.0f},
        {xMax, kFloorTopY, zMin, 0.0f, 0.0f, -1.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 1.0f},

        // Left side (x = xMin)
        {xMin, kFloorBottomY, zMin, -1.0f, 0.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 0.0f},
        {xMin, kFloorBottomY, zMax, -1.0f, 0.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 0.0f},
        {xMin, kFloorTopY, zMax, -1.0f, 0.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 1.0f},
        {xMin, kFloorBottomY, zMin, -1.0f, 0.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 0.0f},
        {xMin, kFloorTopY, zMax, -1.0f, 0.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 1.0f},
        {xMin, kFloorTopY, zMin, -1.0f, 0.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 1.0f},

        // Right side (x = xMax)
        {xMax, kFloorBottomY, zMax, 1.0f, 0.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 0.0f},
        {xMax, kFloorBottomY, zMin, 1.0f, 0.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 0.0f},
        {xMax, kFloorTopY, zMin, 1.0f, 0.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 1.0f},
        {xMax, kFloorBottomY, zMax, 1.0f, 0.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 0.0f},
        {xMax, kFloorTopY, zMin, 1.0f, 0.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 1.0f, 1.0f},
        {xMax, kFloorTopY, zMax, 1.0f, 0.0f, 0.0f, floorColor.x, floorColor.y, floorColor.z, 0.0f, 1.0f},
    };
}

Mat4 buildFloorModelMatrix(CameraMode mode) {
    if (mode == CameraMode::Above) {
        return identityMatrix();
    }

    // Lift the floor block in side/front/back camera modes so it spans the bottom of the viewport.
    return translation(0.0f, 0.24f, 0.0f);
}

std::vector<FlatVertex> buildSnowCap(
    const UmbrellaSilhouette& silhouette,
    const std::array<float, kSnowSamples>& snowLoad
) {
    std::vector<FlatVertex> vertices;
    if (!silhouette.valid) {
        return vertices;
    }

    vertices.reserve((kSnowSamples - 1) * 6);
    for (std::size_t i = 0; i + 1 < kSnowSamples; ++i) {
        const float t0 = static_cast<float>(i) / static_cast<float>(kSnowSamples - 1);
        const float t1 = static_cast<float>(i + 1) / static_cast<float>(kSnowSamples - 1);
        const float x0 = silhouette.xMin + (silhouette.xMax - silhouette.xMin) * t0;
        const float x1 = silhouette.xMin + (silhouette.xMax - silhouette.xMin) * t1;
        const float top0 = silhouette.topY[i];
        const float top1 = silhouette.topY[i + 1];

        if (top0 < -5.0f || top1 < -5.0f) {
            continue;
        }

        const float snowTop0 = top0 + 0.010f + snowLoad[i] * 0.018f;
        const float snowTop1 = top1 + 0.010f + snowLoad[i + 1] * 0.018f;

        vertices.push_back({x0, top0 - 0.002f, 0.0f, 0.97f, 0.98f, 1.0f});
        vertices.push_back({x1, top1 - 0.002f, 0.0f, 0.97f, 0.98f, 1.0f});
        vertices.push_back({x1, snowTop1, 0.0f, 1.0f, 1.0f, 1.0f});

        vertices.push_back({x0, top0 - 0.002f, 0.0f, 0.97f, 0.98f, 1.0f});
        vertices.push_back({x1, snowTop1, 0.0f, 1.0f, 1.0f, 1.0f});
        vertices.push_back({x0, snowTop0, 0.0f, 1.0f, 1.0f, 1.0f});
    }

    return vertices;
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

}  // namespace

int main(int argc, char** argv) {
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
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    glViewport(0, 0, framebufferWidth, framebufferHeight);

    const char* flatVertexShader = R"GLSL(
        #version 330 core
        layout (location = 0) in vec3 aPosition;
        layout (location = 1) in vec3 aColor;

        uniform mat4 uTransform;

        out vec3 vColor;

        void main() {
            gl_Position = uTransform * vec4(aPosition, 1.0);
            vColor = aColor;
        }
    )GLSL";

    const char* flatFragmentShader = R"GLSL(
        #version 330 core
        in vec3 vColor;
        out vec4 FragColor;

        void main() {
            FragColor = vec4(vColor, 1.0);
        }
    )GLSL";

    const char* meshVertexShader = R"GLSL(
        #version 330 core
        layout (location = 0) in vec3 aPosition;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec3 aColor;
        layout (location = 3) in vec2 aTexCoord;

        uniform mat4 uModel;
        uniform bool uFlattenToScreen;

        out vec3 vNormal;
        out vec3 vColor;
        out vec2 vTexCoord;

        void main() {
            vec4 worldPosition = uModel * vec4(aPosition, 1.0);
            if (uFlattenToScreen) {
                gl_Position = vec4(worldPosition.xy, 0.0, 1.0);
            } else {
                gl_Position = vec4(worldPosition.xyz, 1.0);
            }
            vNormal = normalize(mat3(uModel) * aNormal);
            vColor = aColor;
            vTexCoord = aTexCoord;
        }
    )GLSL";

    const char* meshFragmentShader = R"GLSL(
        #version 330 core
        in vec3 vNormal;
        in vec3 vColor;
        in vec2 vTexCoord;

        uniform sampler2D uTexture;
        uniform bool uUseTexture;
        out vec4 FragColor;

        void main() {
            vec3 normal = normalize(vNormal);
            vec3 lightDir = normalize(vec3(-0.35, 0.85, 0.40));

            // Use two-sided diffuse lighting because imported OBJ normals can be inconsistent.
            float diffuse = abs(dot(normal, lightDir));
            float ambient = 0.82;
            float lighting = ambient + 0.18 * diffuse;
            vec3 baseColor = vColor;
            if (uUseTexture) {
                baseColor *= texture(uTexture, vTexCoord).rgb;
            }

            FragColor = vec4(min(baseColor * lighting, vec3(1.0)), 1.0);
        }
    )GLSL";

    const char* particleVertexShader = R"GLSL(
        #version 330 core
        layout (location = 0) in vec3 aPosition;
        layout (location = 1) in float aSize;

        uniform mat4 uView;

        void main() {
            vec4 viewPosition = uView * vec4(aPosition, 1.0);
            gl_Position = viewPosition;

            float depthScale = clamp(1.15 - 0.45 * viewPosition.z, 0.55, 1.35);
            gl_PointSize = aSize * depthScale;
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

    const char* bokehVertexShader = R"GLSL(
        #version 330 core
        layout (location = 0) in vec2 aPosition;
        layout (location = 1) in float aSize;
        layout (location = 2) in float aAlpha;
        layout (location = 3) in float aColorSeed;
        layout (location = 4) in float aTwinkle;
        layout (location = 5) in float aFlicker;

        out float vAlpha;
        out float vColorSeed;
        out float vTwinkle;
        out float vFlicker;

        void main() {
            gl_Position = vec4(aPosition, 0.0, 1.0);
            gl_PointSize = aSize;
            vAlpha = aAlpha;
            vColorSeed = aColorSeed;
            vTwinkle = aTwinkle;
            vFlicker = aFlicker;
        }
    )GLSL";

    const char* bokehFragmentShader = R"GLSL(
        #version 330 core
        in float vAlpha;
        in float vColorSeed;
        in float vTwinkle;
        in float vFlicker;

        uniform float uTime;
        out vec4 FragColor;

        vec3 cityLightColor(float t) {
            vec3 c0 = vec3(1.00, 0.78, 0.52);
            vec3 c1 = vec3(1.00, 0.93, 0.72);
            vec3 c2 = vec3(0.72, 0.84, 1.00);
            vec3 c3 = vec3(0.82, 0.66, 1.00);
            vec3 c4 = vec3(0.62, 1.00, 0.82);

            if (t < 0.24) {
                return mix(c0, c1, t / 0.24);
            }
            if (t < 0.50) {
                return mix(c1, c2, (t - 0.24) / 0.26);
            }
            if (t < 0.74) {
                return mix(c2, c3, (t - 0.50) / 0.24);
            }
            return mix(c3, c4, (t - 0.74) / 0.26);
        }

        void main() {
            vec2 p = gl_PointCoord - vec2(0.5);
            float d = length(p);
            if (d > 0.5) {
                discard;
            }

            float edge = smoothstep(0.52, 0.18, d);
            float core = smoothstep(0.26, 0.0, d);
            float flickerRate = 0.08 + 0.16 * fract(vTwinkle * 0.159);
            float flickerPulse = 0.76 + 0.24 * sin(uTime * flickerRate + vTwinkle);
            float flickerMix = (1.0 - vFlicker) + vFlicker * flickerPulse;
            vec3 color = cityLightColor(vColorSeed) * 0.88;

            FragColor = vec4(color, vAlpha * flickerMix * (0.62 * edge + 0.22 * core));
        }
    )GLSL";

    GLuint flatProgram = createProgram(flatVertexShader, flatFragmentShader);
    GLuint meshProgram = createProgram(meshVertexShader, meshFragmentShader);
    GLuint particleProgram = createProgram(particleVertexShader, particleFragmentShader);
    GLuint bokehProgram = createProgram(bokehVertexShader, bokehFragmentShader);
    if (flatProgram == 0 || meshProgram == 0 || particleProgram == 0 || bokehProgram == 0) {
        glDeleteProgram(flatProgram);
        glDeleteProgram(meshProgram);
        glDeleteProgram(particleProgram);
        glDeleteProgram(bokehProgram);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    std::vector<std::string> musicPaths;
    if (argc > 1 && argv[1] != nullptr) {
        musicPaths.push_back(argv[1]);
    }
    musicPaths.push_back("assets/music.mp3");
    musicPaths.push_back("./assets/music.mp3");
    musicPaths.push_back("../assets/music.mp3");
    musicPaths.push_back("../../assets/music.mp3");

    AudioPlayer* audioPlayer = nullptr;
    const std::string musicPath = findFirstExistingPath(musicPaths);
    if (!musicPath.empty()) {
        std::string audioError;
        audioPlayer = createAudioPlayer(musicPath, audioError);
        if (audioPlayer == nullptr) {
            std::cerr << "Failed to initialize audio from " << musicPath << ": " << audioError << "\n";
        } else if (!startAudioPlayer(audioPlayer, audioError)) {
            std::cerr << "Failed to start audio playback from " << musicPath << ": " << audioError << "\n";
            destroyAudioPlayer(audioPlayer);
            audioPlayer = nullptr;
        } else {
            std::cout << "Playing background audio: " << musicPath << "\n";
        }
    } else {
        std::cerr << "No background audio found. Checked paths:\n";
        for (const std::string& path : musicPaths) {
            std::cerr << "  - " << path << "\n";
        }
    }

    std::vector<std::string> umbrellaPaths = {
        "assets/umbrella/12981_umbrella_v1_l2.obj",
        "./assets/umbrella/12981_umbrella_v1_l2.obj",
        "../assets/umbrella/12981_umbrella_v1_l2.obj",
        "../../assets/umbrella/12981_umbrella_v1_l2.obj"
    };

    UmbrellaMesh umbrellaMesh;
    bool meshLoaded = false;
    for (const std::string& path : umbrellaPaths) {
        if (loadUmbrellaMesh(path, umbrellaMesh)) {
            meshLoaded = true;
            break;
        }
    }

    if (!meshLoaded) {
        std::cerr << "Failed to load umbrella OBJ. Checked paths:\n";
        for (const std::string& path : umbrellaPaths) {
            std::error_code ec;
            const bool exists = std::filesystem::exists(path, ec);
            std::cerr << "  - " << path << (exists ? " (exists)" : " (missing)") << "\n";
        }
        glDeleteProgram(flatProgram);
        glDeleteProgram(meshProgram);
        glDeleteProgram(particleProgram);
        glDeleteProgram(bokehProgram);
        destroyAudioPlayer(audioPlayer);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    const Mat4 umbrellaModel = buildUmbrellaModelMatrix(umbrellaMesh.bounds);
    const UmbrellaSilhouette silhouette = buildSilhouette(umbrellaMesh.canopyTriangles, umbrellaModel);
    const UmbrellaHeightField umbrellaHeightField = buildUmbrellaHeightField(umbrellaMesh.canopyTriangles, umbrellaModel);

    std::vector<std::string> characterPaths = {
        "assets/im_sol_arm_out.glb",
        "./assets/im_sol_arm_out.glb",
        "../assets/im_sol_arm_out.glb",
        "../../assets/im_sol_arm_out.glb"
    };

    SceneMesh characterMesh;
    TextureImage characterTextureImage;
    bool characterLoaded = false;
    for (const std::string& path : characterPaths) {
        if (loadGlbMesh(path, characterMesh, characterTextureImage)) {
            characterLoaded = true;
            break;
        }
    }

    if (!characterLoaded) {
        std::cerr << "Failed to load im_sol_arm_out.glb. Checked paths:\n";
        for (const std::string& path : characterPaths) {
            std::error_code ec;
            const bool exists = std::filesystem::exists(path, ec);
            std::cerr << "  - " << path << (exists ? " (exists)" : " (missing)") << "\n";
        }
        glDeleteProgram(flatProgram);
        glDeleteProgram(meshProgram);
        glDeleteProgram(particleProgram);
        glDeleteProgram(bokehProgram);
        destroyAudioPlayer(audioPlayer);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    const Mat4 characterModel = buildCharacterModelMatrix(characterMesh.bounds);

    std::vector<std::string> sunjaePaths = {
        "assets/sunjae.glb",
        "./assets/sunjae.glb",
        "../assets/sunjae.glb",
        "../../assets/sunjae.glb"
    };

    SceneMesh sunjaeMesh;
    TextureImage sunjaeTextureImage;
    bool sunjaeLoaded = false;
    for (const std::string& path : sunjaePaths) {
        if (loadGlbMesh(path, sunjaeMesh, sunjaeTextureImage)) {
            sunjaeLoaded = true;
            break;
        }
    }

    if (!sunjaeLoaded) {
        std::cerr << "Failed to load sunjae.glb. Checked paths:\n";
        for (const std::string& path : sunjaePaths) {
            std::error_code ec;
            const bool exists = std::filesystem::exists(path, ec);
            std::cerr << "  - " << path << (exists ? " (exists)" : " (missing)") << "\n";
        }
        glDeleteProgram(flatProgram);
        glDeleteProgram(meshProgram);
        glDeleteProgram(particleProgram);
        glDeleteProgram(bokehProgram);
        destroyAudioPlayer(audioPlayer);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    const Mat4 sunjaeModel = buildSunjaeModelMatrix(sunjaeMesh.bounds);
    GLuint characterTexture = createTexture2D(characterTextureImage);
    GLuint sunjaeTexture = createTexture2D(sunjaeTextureImage);
    GLuint whiteFallbackTexture = createSolidWhiteTexture();

    GLuint backgroundVao = 0;
    GLuint backgroundVbo = 0;
    glGenVertexArrays(1, &backgroundVao);
    glGenBuffers(1, &backgroundVbo);

    const std::vector<FlatVertex> background = buildBackground();
    glBindVertexArray(backgroundVao);
    glBindBuffer(GL_ARRAY_BUFFER, backgroundVbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(background.size() * sizeof(FlatVertex)), background.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(FlatVertex), reinterpret_cast<void*>(offsetof(FlatVertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(FlatVertex), reinterpret_cast<void*>(offsetof(FlatVertex, r)));
    glEnableVertexAttribArray(1);

    GLuint groundBandVao = 0;
    GLuint groundBandVbo = 0;
    glGenVertexArrays(1, &groundBandVao);
    glGenBuffers(1, &groundBandVbo);

    const std::vector<FlatVertex> groundBand = buildGroundBand();
    glBindVertexArray(groundBandVao);
    glBindBuffer(GL_ARRAY_BUFFER, groundBandVbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(groundBand.size() * sizeof(FlatVertex)), groundBand.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(FlatVertex), reinterpret_cast<void*>(offsetof(FlatVertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(FlatVertex), reinterpret_cast<void*>(offsetof(FlatVertex, r)));
    glEnableVertexAttribArray(1);

    GLuint topFloorFillVao = 0;
    GLuint topFloorFillVbo = 0;
    glGenVertexArrays(1, &topFloorFillVao);
    glGenBuffers(1, &topFloorFillVbo);

    const std::vector<FlatVertex> topFloorFill = buildTopViewFloorFill();
    glBindVertexArray(topFloorFillVao);
    glBindBuffer(GL_ARRAY_BUFFER, topFloorFillVbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(topFloorFill.size() * sizeof(FlatVertex)), topFloorFill.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(FlatVertex), reinterpret_cast<void*>(offsetof(FlatVertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(FlatVertex), reinterpret_cast<void*>(offsetof(FlatVertex, r)));
    glEnableVertexAttribArray(1);

    GLuint floorVao = 0;
    GLuint floorVbo = 0;
    glGenVertexArrays(1, &floorVao);
    glGenBuffers(1, &floorVbo);

    const std::vector<MeshVertex> floorMesh = buildFloorMesh();
    glBindVertexArray(floorVao);
    glBindBuffer(GL_ARRAY_BUFFER, floorVbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(floorMesh.size() * sizeof(MeshVertex)),
        floorMesh.data(),
        GL_STATIC_DRAW
    );
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, nx)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, r)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, u)));
    glEnableVertexAttribArray(3);

    GLuint sunjaeVao = 0;
    GLuint sunjaeVbo = 0;
    glGenVertexArrays(1, &sunjaeVao);
    glGenBuffers(1, &sunjaeVbo);

    glBindVertexArray(sunjaeVao);
    glBindBuffer(GL_ARRAY_BUFFER, sunjaeVbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(sunjaeMesh.vertices.size() * sizeof(MeshVertex)),
        sunjaeMesh.vertices.data(),
        GL_STATIC_DRAW
    );
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, nx)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, r)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, u)));
    glEnableVertexAttribArray(3);

    GLuint umbrellaVao = 0;
    GLuint umbrellaVbo = 0;
    glGenVertexArrays(1, &umbrellaVao);
    glGenBuffers(1, &umbrellaVbo);

    glBindVertexArray(umbrellaVao);
    glBindBuffer(GL_ARRAY_BUFFER, umbrellaVbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(umbrellaMesh.vertices.size() * sizeof(MeshVertex)),
        umbrellaMesh.vertices.data(),
        GL_STATIC_DRAW
    );
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, nx)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, r)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, u)));
    glEnableVertexAttribArray(3);

    GLuint characterVao = 0;
    GLuint characterVbo = 0;
    glGenVertexArrays(1, &characterVao);
    glGenBuffers(1, &characterVbo);

    glBindVertexArray(characterVao);
    glBindBuffer(GL_ARRAY_BUFFER, characterVbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(characterMesh.vertices.size() * sizeof(MeshVertex)),
        characterMesh.vertices.data(),
        GL_STATIC_DRAW
    );
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, nx)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, r)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, u)));
    glEnableVertexAttribArray(3);

    GLuint snowCapVao = 0;
    GLuint snowCapVbo = 0;
    glGenVertexArrays(1, &snowCapVao);
    glGenBuffers(1, &snowCapVbo);

    glBindVertexArray(snowCapVao);
    glBindBuffer(GL_ARRAY_BUFFER, snowCapVbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>((kSnowSamples - 1) * 6 * sizeof(FlatVertex)), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(FlatVertex), reinterpret_cast<void*>(offsetof(FlatVertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(FlatVertex), reinterpret_cast<void*>(offsetof(FlatVertex, r)));
    glEnableVertexAttribArray(1);

    GLuint particleVao = 0;
    GLuint particleVbo = 0;
    glGenVertexArrays(1, &particleVao);
    glGenBuffers(1, &particleVbo);

    std::vector<Particle> particles;
    particles.reserve(kParticleCount);

    std::vector<BokehParticle> bokehParticles;
    bokehParticles.reserve(kBokehCount);

    std::random_device rd;
    std::mt19937 rng(rd());
    for (std::size_t i = 0; i < kParticleCount; ++i) {
        particles.push_back(makeParticle(rng, false));
    }
    for (std::size_t i = 0; i < kBokehCount; ++i) {
        bokehParticles.push_back(makeBokehParticle(rng, false));
    }

    std::vector<ParticleVertex> particleVertices(kParticleCount);
    glBindVertexArray(particleVao);
    glBindBuffer(GL_ARRAY_BUFFER, particleVbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(particleVertices.size() * sizeof(ParticleVertex)),
        nullptr,
        GL_DYNAMIC_DRAW
    );
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), reinterpret_cast<void*>(offsetof(ParticleVertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), reinterpret_cast<void*>(offsetof(ParticleVertex, size)));
    glEnableVertexAttribArray(1);

    GLuint bokehVao = 0;
    GLuint bokehVbo = 0;
    glGenVertexArrays(1, &bokehVao);
    glGenBuffers(1, &bokehVbo);

    std::vector<BokehVertex> bokehVertices(kBokehCount);
    glBindVertexArray(bokehVao);
    glBindBuffer(GL_ARRAY_BUFFER, bokehVbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(bokehVertices.size() * sizeof(BokehVertex)),
        nullptr,
        GL_DYNAMIC_DRAW
    );
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(BokehVertex), reinterpret_cast<void*>(offsetof(BokehVertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(BokehVertex), reinterpret_cast<void*>(offsetof(BokehVertex, size)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(BokehVertex), reinterpret_cast<void*>(offsetof(BokehVertex, alpha)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(BokehVertex), reinterpret_cast<void*>(offsetof(BokehVertex, colorSeed)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(BokehVertex), reinterpret_cast<void*>(offsetof(BokehVertex, twinkle)));
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, sizeof(BokehVertex), reinterpret_cast<void*>(offsetof(BokehVertex, flicker)));
    glEnableVertexAttribArray(5);

    GLuint settledSnowVao = 0;
    GLuint settledSnowVbo = 0;
    glGenVertexArrays(1, &settledSnowVao);
    glGenBuffers(1, &settledSnowVbo);

    std::vector<ParticleVertex> settledSnowVertices(UmbrellaHeightField::kSamples * UmbrellaHeightField::kSamples);
    glBindVertexArray(settledSnowVao);
    glBindBuffer(GL_ARRAY_BUFFER, settledSnowVbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(settledSnowVertices.size() * sizeof(ParticleVertex)),
        nullptr,
        GL_DYNAMIC_DRAW
    );
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), reinterpret_cast<void*>(offsetof(ParticleVertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), reinterpret_cast<void*>(offsetof(ParticleVertex, size)));
    glEnableVertexAttribArray(1);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    std::array<float, kSnowSamples> snowLoad{};
    std::array<float, UmbrellaHeightField::kSamples * UmbrellaHeightField::kSamples> settledSnowLoad{};
    float lastTime = static_cast<float>(glfwGetTime());
    CameraMode cameraMode = CameraMode::Front;
    bool spaceWasDown = false;

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

        const bool spaceDown = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spaceDown && !spaceWasDown) {
            const int nextMode = (static_cast<int>(cameraMode) + 1) % static_cast<int>(CameraMode::Count);
            cameraMode = static_cast<CameraMode>(nextMode);
        }
        spaceWasDown = spaceDown;

        for (float& sample : snowLoad) {
            sample = std::max(0.0f, sample - deltaTime * 0.025f);
        }

        for (float& sample : settledSnowLoad) {
            sample = std::max(0.0f, sample - deltaTime * 0.010f);
        }

        for (std::size_t i = 0; i < particles.size(); ++i) {
            Particle& particle = particles[i];
            particle.y -= particle.speed * deltaTime;
            particle.x += std::sin(currentTime * particle.drift + particle.phase) * 0.16f * deltaTime;
            particle.z += std::cos(currentTime * particle.drift * 0.8f + particle.phase) * 0.06f * deltaTime;

            if (hitsUmbrella3D(umbrellaHeightField, particle.x, particle.y, particle.z)) {
                if (umbrellaHeightField.valid &&
                    particle.x >= umbrellaHeightField.xMin && particle.x <= umbrellaHeightField.xMax &&
                    particle.z >= umbrellaHeightField.zMin && particle.z <= umbrellaHeightField.zMax) {
                    const float nx = (particle.x - umbrellaHeightField.xMin) / (umbrellaHeightField.xMax - umbrellaHeightField.xMin);
                    const float nz = (particle.z - umbrellaHeightField.zMin) / (umbrellaHeightField.zMax - umbrellaHeightField.zMin);
                    const std::size_t xi = std::min(
                        static_cast<std::size_t>(std::clamp(nx, 0.0f, 0.999f) * static_cast<float>(UmbrellaHeightField::kSamples)),
                        UmbrellaHeightField::kSamples - 1
                    );
                    const std::size_t zi = std::min(
                        static_cast<std::size_t>(std::clamp(nz, 0.0f, 0.999f) * static_cast<float>(UmbrellaHeightField::kSamples)),
                        UmbrellaHeightField::kSamples - 1
                    );
                    const std::size_t index = zi * UmbrellaHeightField::kSamples + xi;
                    settledSnowLoad[index] = std::min(1.0f, settledSnowLoad[index] + 0.18f);
                }
                particle = makeParticle(rng, true);
            } else if (
                particle.y < -1.1f ||
                particle.x < -1.15f || particle.x > 1.15f ||
                particle.z < -1.15f || particle.z > 1.15f
            ) {
                particle = makeParticle(rng, true);
            }

            particleVertices[i] = {particle.x, particle.y, particle.z, particle.size};
        }

        for (std::size_t i = 0; i < bokehParticles.size(); ++i) {
            BokehParticle& particle = bokehParticles[i];

            bokehVertices[i] = {
                particle.x,
                particle.y,
                particle.size,
                particle.alpha,
                particle.colorSeed,
                particle.twinkle,
                particle.flicker
            };
        }

        std::size_t settledCount = 0;
        for (std::size_t zi = 0; zi < UmbrellaHeightField::kSamples; ++zi) {
            const float tz = static_cast<float>(zi) / static_cast<float>(UmbrellaHeightField::kSamples - 1);
            const float z = umbrellaHeightField.zMin + (umbrellaHeightField.zMax - umbrellaHeightField.zMin) * tz;
            for (std::size_t xi = 0; xi < UmbrellaHeightField::kSamples; ++xi) {
                const std::size_t index = zi * UmbrellaHeightField::kSamples + xi;
                const float load = settledSnowLoad[index];
                if (load < 0.04f) {
                    continue;
                }

                const float tx = static_cast<float>(xi) / static_cast<float>(UmbrellaHeightField::kSamples - 1);
                const float x = umbrellaHeightField.xMin + (umbrellaHeightField.xMax - umbrellaHeightField.xMin) * tx;
                const float y = umbrellaHeightField.topY[index] + 0.006f + load * 0.03f;

                settledSnowVertices[settledCount++] = {
                    x,
                    y,
                    z,
                    2.0f + load * 5.0f
                };
            }
        }

        const std::vector<FlatVertex> snowCap = buildSnowCap(silhouette, snowLoad);

        glBindBuffer(GL_ARRAY_BUFFER, snowCapVbo);
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            static_cast<GLsizeiptr>(snowCap.size() * sizeof(FlatVertex)),
            snowCap.data()
        );

        glBindBuffer(GL_ARRAY_BUFFER, particleVbo);
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            static_cast<GLsizeiptr>(particleVertices.size() * sizeof(ParticleVertex)),
            particleVertices.data()
        );

        glBindBuffer(GL_ARRAY_BUFFER, settledSnowVbo);
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            static_cast<GLsizeiptr>(settledCount * sizeof(ParticleVertex)),
            settledSnowVertices.data()
        );

        glBindBuffer(GL_ARRAY_BUFFER, bokehVbo);
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            static_cast<GLsizeiptr>(bokehVertices.size() * sizeof(BokehVertex)),
            bokehVertices.data()
        );

        glClearColor(0.03f, 0.05f, 0.14f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const Mat4 cameraView = buildCameraViewTransform(cameraMode);

        glDisable(GL_DEPTH_TEST);
        glUseProgram(flatProgram);
        const Mat4 flatIdentity = identityMatrix();
        glUniformMatrix4fv(glGetUniformLocation(flatProgram, "uTransform"), 1, GL_FALSE, flatIdentity.m.data());
        glBindVertexArray(backgroundVao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(background.size()));

        if (cameraMode == CameraMode::Above) {
            glBindVertexArray(topFloorFillVao);
            glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(topFloorFill.size()));
        } else {
            glBindVertexArray(groundBandVao);
            glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(groundBand.size()));
        }

        // Draw bokeh in the background layer so scene geometry stays in front.
        glUseProgram(bokehProgram);
        glUniform1f(glGetUniformLocation(bokehProgram, "uTime"), currentTime);
        glBindVertexArray(bokehVao);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(bokehVertices.size()));

        glEnable(GL_DEPTH_TEST);
        if (cameraMode != CameraMode::Above) {
            glUseProgram(meshProgram);
            glUniform1i(glGetUniformLocation(meshProgram, "uFlattenToScreen"), 0);
            glUniform1i(glGetUniformLocation(meshProgram, "uUseTexture"), 0);
            const Mat4 floorModel = buildFloorModelMatrix(cameraMode);
            const Mat4 displayedFloorModel = multiply(cameraView, floorModel);
            glUniformMatrix4fv(glGetUniformLocation(meshProgram, "uModel"), 1, GL_FALSE, displayedFloorModel.m.data());
            glBindVertexArray(floorVao);
            glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(floorMesh.size()));
        }

        glUseProgram(meshProgram);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, whiteFallbackTexture);
        glUniform1i(glGetUniformLocation(meshProgram, "uTexture"), 0);
        glUniform1i(glGetUniformLocation(meshProgram, "uFlattenToScreen"), 0);
        glUniform1i(glGetUniformLocation(meshProgram, "uUseTexture"), 0);
        const Mat4 displayedUmbrellaModel = multiply(cameraView, umbrellaModel);
        glUniformMatrix4fv(glGetUniformLocation(meshProgram, "uModel"), 1, GL_FALSE, displayedUmbrellaModel.m.data());
        glBindVertexArray(umbrellaVao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(umbrellaMesh.vertices.size()));

        glUseProgram(meshProgram);
        glUniform1i(glGetUniformLocation(meshProgram, "uFlattenToScreen"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, characterTexture != 0 ? characterTexture : whiteFallbackTexture);
        glUniform1i(glGetUniformLocation(meshProgram, "uTexture"), 0);
        glUniform1i(glGetUniformLocation(meshProgram, "uUseTexture"), characterTexture != 0 ? 1 : 0);
        const Mat4 displayedCharacterModel = multiply(cameraView, characterModel);
        glUniformMatrix4fv(glGetUniformLocation(meshProgram, "uModel"), 1, GL_FALSE, displayedCharacterModel.m.data());
        glBindVertexArray(characterVao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(characterMesh.vertices.size()));

        glBindTexture(GL_TEXTURE_2D, sunjaeTexture != 0 ? sunjaeTexture : whiteFallbackTexture);
        glUniform1i(glGetUniformLocation(meshProgram, "uUseTexture"), sunjaeTexture != 0 ? 1 : 0);
        const Mat4 displayedSunjaeModel = multiply(cameraView, sunjaeModel);
        glUniformMatrix4fv(glGetUniformLocation(meshProgram, "uModel"), 1, GL_FALSE, displayedSunjaeModel.m.data());
        glBindVertexArray(sunjaeVao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(sunjaeMesh.vertices.size()));
        glEnable(GL_DEPTH_TEST);
        glUseProgram(particleProgram);
        glUniformMatrix4fv(glGetUniformLocation(particleProgram, "uView"), 1, GL_FALSE, cameraView.m.data());

        glBindVertexArray(settledSnowVao);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(settledCount));

        glBindVertexArray(particleVao);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particleVertices.size()));

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteBuffers(1, &bokehVbo);
    glDeleteVertexArrays(1, &bokehVao);
    glDeleteBuffers(1, &particleVbo);
    glDeleteVertexArrays(1, &particleVao);
    glDeleteBuffers(1, &settledSnowVbo);
    glDeleteVertexArrays(1, &settledSnowVao);
    glDeleteBuffers(1, &snowCapVbo);
    glDeleteVertexArrays(1, &snowCapVao);
    glDeleteTextures(1, &sunjaeTexture);
    glDeleteTextures(1, &characterTexture);
    glDeleteTextures(1, &whiteFallbackTexture);
    glDeleteBuffers(1, &topFloorFillVbo);
    glDeleteVertexArrays(1, &topFloorFillVao);
    glDeleteBuffers(1, &groundBandVbo);
    glDeleteVertexArrays(1, &groundBandVao);
    glDeleteBuffers(1, &floorVbo);
    glDeleteVertexArrays(1, &floorVao);
    glDeleteBuffers(1, &sunjaeVbo);
    glDeleteVertexArrays(1, &sunjaeVao);
    glDeleteBuffers(1, &characterVbo);
    glDeleteVertexArrays(1, &characterVao);
    glDeleteBuffers(1, &umbrellaVbo);
    glDeleteVertexArrays(1, &umbrellaVao);
    glDeleteBuffers(1, &backgroundVbo);
    glDeleteVertexArrays(1, &backgroundVao);
    glDeleteProgram(flatProgram);
    glDeleteProgram(meshProgram);
    glDeleteProgram(particleProgram);
    glDeleteProgram(bokehProgram);
    destroyAudioPlayer(audioPlayer);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
