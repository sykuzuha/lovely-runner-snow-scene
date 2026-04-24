#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <png.h>

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

constexpr int kWindowWidth = 960;
constexpr int kWindowHeight = 640;
constexpr std::size_t kParticleCount = 900;
constexpr std::size_t kSnowSamples = 48;

std::string trim(const std::string& value) {
    const std::size_t start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }

    const std::size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
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

    const float extentX = bounds.max.x - bounds.min.x;
    const float extentY = bounds.max.y - bounds.min.y;
    const float extentZ = bounds.max.z - bounds.min.z;
    const float maxExtent = std::max({extentX, extentY, extentZ});
    const float scale = 0.0185f;

    Mat4 model = identityMatrix();
    model = multiply(translation(0.18f, 0.34f, 0.12f), model);
    model = multiply(rotationZ(-0.18f), model);
    model = multiply(rotationY(0.12f), model);
    model = multiply(rotationX(-1.57f), model);
    model = multiply(uniformScale(scale * (37.5f / maxExtent)), model);
    model = multiply(translation(-center.x, -center.y, -center.z), model);
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

    Mat4 model = identityMatrix();
    model = multiply(rotationY(-1.57f), model);
    model = multiply(scale(1.0f, 1.0f, -1.0f), model);
    model = multiply(uniformScale(normalizedScale), model);
    model = multiply(translation(-center.x, -center.y, -center.z), model);
    // Apply final placement in world space so X/Y/Z edits move the model predictably on screen.
    model = multiply(translation(0.12f, -0.18f, -0.04f), model);
    return model;
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

float silhouetteTopY(const UmbrellaSilhouette& silhouette, float x) {
    if (!silhouette.valid || x < silhouette.xMin || x > silhouette.xMax) {
        return -10.0f;
    }

    const float normalized = (x - silhouette.xMin) / (silhouette.xMax - silhouette.xMin);
    const float position = normalized * static_cast<float>(kSnowSamples - 1);
    const std::size_t index = static_cast<std::size_t>(position);
    const std::size_t nextIndex = std::min(index + 1, kSnowSamples - 1);
    const float alpha = position - static_cast<float>(index);
    return silhouette.topY[index] + (silhouette.topY[nextIndex] - silhouette.topY[index]) * alpha;
}

bool hitsUmbrella(const UmbrellaSilhouette& silhouette, float x, float y) {
    const float topY = silhouetteTopY(silhouette, x);
    return topY > -5.0f && y <= topY + 0.008f && y >= topY - 0.05f;
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

std::vector<FlatVertex> buildBackground() {
    return {
        {-1.0f, -1.0f, 0.0f, 0.86f, 0.89f, 0.96f},
        { 1.0f, -1.0f, 0.0f, 0.86f, 0.89f, 0.96f},
        { 1.0f, -0.50f, 0.0f, 0.76f, 0.81f, 0.92f},
        {-1.0f, -1.0f, 0.0f, 0.86f, 0.89f, 0.96f},
        { 1.0f, -0.50f, 0.0f, 0.76f, 0.81f, 0.92f},
        {-1.0f, -0.50f, 0.0f, 0.76f, 0.81f, 0.92f},
    };
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
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    glViewport(0, 0, framebufferWidth, framebufferHeight);

    const char* flatVertexShader = R"GLSL(
        #version 330 core
        layout (location = 0) in vec3 aPosition;
        layout (location = 1) in vec3 aColor;

        out vec3 vColor;

        void main() {
            gl_Position = vec4(aPosition, 1.0);
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

    GLuint flatProgram = createProgram(flatVertexShader, flatFragmentShader);
    GLuint meshProgram = createProgram(meshVertexShader, meshFragmentShader);
    GLuint particleProgram = createProgram(particleVertexShader, particleFragmentShader);
    if (flatProgram == 0 || meshProgram == 0 || particleProgram == 0) {
        glDeleteProgram(flatProgram);
        glDeleteProgram(meshProgram);
        glDeleteProgram(particleProgram);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
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
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    const Mat4 umbrellaModel = buildUmbrellaModelMatrix(umbrellaMesh.bounds);
    const UmbrellaSilhouette silhouette = buildSilhouette(umbrellaMesh.canopyTriangles, umbrellaModel);

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
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    const Mat4 characterModel = buildCharacterModelMatrix(characterMesh.bounds);
    GLuint characterTexture = createTexture2D(characterTextureImage);
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

    std::random_device rd;
    std::mt19937 rng(rd());
    for (std::size_t i = 0; i < kParticleCount; ++i) {
        particles.push_back(makeParticle(rng, false));
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
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), reinterpret_cast<void*>(offsetof(ParticleVertex, x)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), reinterpret_cast<void*>(offsetof(ParticleVertex, size)));
    glEnableVertexAttribArray(1);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    std::array<float, kSnowSamples> snowLoad{};
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

        for (float& sample : snowLoad) {
            sample = std::max(0.0f, sample - deltaTime * 0.025f);
        }

        for (std::size_t i = 0; i < particles.size(); ++i) {
            Particle& particle = particles[i];
            particle.y -= particle.speed * deltaTime;
            particle.x += std::sin(currentTime * particle.drift + particle.phase) * 0.16f * deltaTime;

            if (hitsUmbrella(silhouette, particle.x, particle.y)) {
                const float normalized = (particle.x - silhouette.xMin) / (silhouette.xMax - silhouette.xMin);
                const std::size_t sampleIndex = std::min(
                    static_cast<std::size_t>(std::clamp(normalized, 0.0f, 0.999f) * static_cast<float>(kSnowSamples)),
                    kSnowSamples - 1
                );
                snowLoad[sampleIndex] = std::min(1.0f, snowLoad[sampleIndex] + 0.16f);
                particle = makeParticle(rng, true);
            } else if (particle.y < -1.1f || particle.x < -1.15f || particle.x > 1.15f) {
                particle = makeParticle(rng, true);
            }

            particleVertices[i] = {particle.x, particle.y, particle.size};
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

        glClearColor(0.56f, 0.66f, 0.82f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDisable(GL_DEPTH_TEST);
        glUseProgram(flatProgram);
        glBindVertexArray(backgroundVao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(background.size()));

        glEnable(GL_DEPTH_TEST);
        glUseProgram(meshProgram);
        glUniform1i(glGetUniformLocation(meshProgram, "uFlattenToScreen"), 1);
        glUniform1i(glGetUniformLocation(meshProgram, "uUseTexture"), 0);
        glUniformMatrix4fv(glGetUniformLocation(meshProgram, "uModel"), 1, GL_FALSE, umbrellaModel.m.data());
        glBindVertexArray(umbrellaVao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(umbrellaMesh.vertices.size()));

        glUseProgram(meshProgram);
        glUniform1i(glGetUniformLocation(meshProgram, "uFlattenToScreen"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, characterTexture != 0 ? characterTexture : whiteFallbackTexture);
        glUniform1i(glGetUniformLocation(meshProgram, "uTexture"), 0);
        glUniform1i(glGetUniformLocation(meshProgram, "uUseTexture"), characterTexture != 0 ? 1 : 0);
        glUniformMatrix4fv(glGetUniformLocation(meshProgram, "uModel"), 1, GL_FALSE, characterModel.m.data());
        glBindVertexArray(characterVao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(characterMesh.vertices.size()));
        glBindTexture(GL_TEXTURE_2D, 0);

        glDisable(GL_DEPTH_TEST);
        glUseProgram(flatProgram);
        glBindVertexArray(snowCapVao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(snowCap.size()));

        glUseProgram(particleProgram);
        glBindVertexArray(particleVao);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particleVertices.size()));

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteBuffers(1, &particleVbo);
    glDeleteVertexArrays(1, &particleVao);
    glDeleteBuffers(1, &snowCapVbo);
    glDeleteVertexArrays(1, &snowCapVao);
    glDeleteTextures(1, &characterTexture);
    glDeleteTextures(1, &whiteFallbackTexture);
    glDeleteBuffers(1, &characterVbo);
    glDeleteVertexArrays(1, &characterVao);
    glDeleteBuffers(1, &umbrellaVbo);
    glDeleteVertexArrays(1, &umbrellaVao);
    glDeleteBuffers(1, &backgroundVbo);
    glDeleteVertexArrays(1, &backgroundVao);
    glDeleteProgram(flatProgram);
    glDeleteProgram(meshProgram);
    glDeleteProgram(particleProgram);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
