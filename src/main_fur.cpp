// main_fur.cpp — Cat with fur + snow scene
// Merges cat_scene (snow/umbrella/background) with cat_fur (tinygltf + fur rendering).
// Dependencies: tinygltf (header-only), glm, glad, glfw — no assimp/libpng needed.

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

// ─── Window / scene constants ────────────────────────────────────────────────
const int W = 960;
const int H = 640;

// ─── Fur constants ───────────────────────────────────────────────────────────
const int   STRANDS_PER_FACE  = 20;
const int   SEGMENTS          = 14;
const float BASE_LENGTH       = 0.036f;
const float CURL_AMOUNT       = 0.12f;
const float LENGTH_VARIANCE   = 0.2f;
const float CURL_FREQ         = 1.0f;
const int   RANDOM_SEED       = 42;

// ─── Snow / scene constants ───────────────────────────────────────────────────
constexpr std::size_t kParticleCount = 900;
constexpr std::size_t kSnowSamples   = 48;
constexpr float kFloorTopY    = -0.42f;
constexpr float kFloorBottomY = -0.90f;

// ─── Camera orbit (for the fur/3D view) ──────────────────────────────────────
float orbitYaw   = 180.0f;
float orbitPitch = 6.0f;
float orbitDist  = 4.5f;
glm::vec3 orbitTarget(0.0f);
double lastX = W / 2.0;
double lastY = H / 2.0;
bool mouseDown = false;

// ─── Camera mode (space to cycle views) ──────────────────────────────────────
enum class CameraMode { Front = 0, Left, Right, Back, Above, Count };
CameraMode cameraMode = CameraMode::Front;
bool spaceWasDown = false;

void framebuffer_size_callback(GLFWwindow*, int w, int h) { glViewport(0, 0, w, h); }

void mouse_button_callback(GLFWwindow*, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) mouseDown = (action == GLFW_PRESS);
}

void cursor_callback(GLFWwindow*, double xpos, double ypos) {
    double dx = xpos - lastX;
    double dy = ypos - lastY;
    lastX = xpos; lastY = ypos;
    if (!mouseDown) return;
    orbitYaw   += static_cast<float>(dx) * 0.4f;
    orbitPitch  = glm::clamp(orbitPitch - static_cast<float>(dy) * 0.4f, -89.0f, 89.0f);
}

void scroll_callback(GLFWwindow*, double, double yoff) {
    orbitDist = glm::clamp(orbitDist - static_cast<float>(yoff) * 0.2f, 0.35f, 20.0f);
}

// ─── Shader helpers ───────────────────────────────────────────────────────────
GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetShaderInfoLog(shader, sizeof(buf), nullptr, buf);
        std::cerr << "Shader error: " << buf << "\n";
    }
    return shader;
}

GLuint makeProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetProgramInfoLog(p, sizeof(buf), nullptr, buf);
        std::cerr << "Program link error: " << buf << "\n";
    }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// ─── GLB / mesh data structures ──────────────────────────────────────────────
struct ImageData {
    int width = 0, height = 0, channels = 0;
    std::vector<unsigned char> pixels;
};

struct MaterialData {
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    int textureIndex = -1;
    bool doubleSided = false;
};

struct SubMesh {
    std::string name;
    std::vector<glm::vec3> positions, normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::ivec3> faces;
    int materialIndex = -1;
};

struct SceneData {
    std::vector<SubMesh>      submeshes;
    std::vector<MaterialData> materials;
    std::vector<ImageData>    images;
    glm::vec3 boundsMin = glm::vec3( std::numeric_limits<float>::max());
    glm::vec3 boundsMax = glm::vec3(-std::numeric_limits<float>::max());
};

// ─── Snow / scene data structures ─────────────────────────────────────────────
struct Vec3s { float x, y, z; };
struct FlatVertex  { float x, y, z, r, g, b; };
struct MeshVertex3 { float x, y, z, nx, ny, nz, r, g, b, u, v; };
struct ParticleVertex { float x, y, z, size; };
struct Particle { float x, y, z, speed, drift, phase, size; };

struct BokehParticle { float x, y, speed, drift, phase, size, alpha, colorSeed, twinkle, flicker; };
struct BokehVertex   { float x, y, size, alpha, colorSeed, twinkle, flicker; };

constexpr std::size_t kBokehCount = 56;

struct UmbrellaMesh {
    std::vector<MeshVertex3> vertices;
    std::vector<Vec3s>       canopyTriangles;
    glm::vec3 boundsMin = glm::vec3( std::numeric_limits<float>::max());
    glm::vec3 boundsMax = glm::vec3(-std::numeric_limits<float>::max());
};

struct UmbrellaSilhouette {
    bool  valid = false;
    float xMin = 0, xMax = 0;
    std::array<float, kSnowSamples> topY{};
};

struct UmbrellaHeightField {
    static constexpr std::size_t kSamples = 40;
    bool  valid = false;
    float xMin = 0, xMax = 0, zMin = 0, zMax = 0;
    std::array<float, kSamples * kSamples> topY{};
};

// ─── GLB loading (tinygltf) ──────────────────────────────────────────────────
glm::mat4 nodeLocalMatrix(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) return glm::make_mat4(node.matrix.data());
    glm::vec3 t(0.0f); glm::vec3 s(1.0f); glm::quat r(1,0,0,0);
    if (node.translation.size()==3) t = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
    if (node.scale.size()==3)       s = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
    if (node.rotation.size()==4)    r = glm::quat(static_cast<float>(node.rotation[3]),
                                                   static_cast<float>(node.rotation[0]),
                                                   static_cast<float>(node.rotation[1]),
                                                   static_cast<float>(node.rotation[2]));
    return glm::translate(glm::mat4(1), t) * glm::mat4_cast(r) * glm::scale(glm::mat4(1), s);
}

size_t componentSize(int t) {
    switch(t){
        case TINYGLTF_COMPONENT_TYPE_BYTE: case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: return 1;
        case TINYGLTF_COMPONENT_TYPE_SHORT: case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: return 2;
        case TINYGLTF_COMPONENT_TYPE_INT: case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        case TINYGLTF_COMPONENT_TYPE_FLOAT: return 4;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE: return 8;
        default: return 0;
    }
}

std::vector<glm::vec3> readVec3(const tinygltf::Model& m, int idx) {
    auto& acc = m.accessors[idx]; auto& view = m.bufferViews[acc.bufferView];
    auto& buf = m.buffers[view.buffer];
    const unsigned char* data = buf.data.data() + view.byteOffset + acc.byteOffset;
    size_t stride = acc.ByteStride(view) ? acc.ByteStride(view) : 3*sizeof(float);
    std::vector<glm::vec3> out(acc.count);
    for (size_t i=0;i<acc.count;++i){ const float* s=reinterpret_cast<const float*>(data+i*stride); out[i]={s[0],s[1],s[2]}; }
    return out;
}

std::vector<glm::vec2> readVec2(const tinygltf::Model& m, int idx) {
    auto& acc = m.accessors[idx]; auto& view = m.bufferViews[acc.bufferView];
    auto& buf = m.buffers[view.buffer];
    const unsigned char* data = buf.data.data() + view.byteOffset + acc.byteOffset;
    size_t stride = acc.ByteStride(view) ? acc.ByteStride(view) : 2*sizeof(float);
    std::vector<glm::vec2> out(acc.count, glm::vec2(0));
    for (size_t i=0;i<acc.count;++i){ const float* s=reinterpret_cast<const float*>(data+i*stride); out[i]={s[0],s[1]}; }
    return out;
}

uint32_t readIdx(const unsigned char* ptr, int t) {
    switch(t){
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  return *reinterpret_cast<const uint8_t*>(ptr);
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: return *reinterpret_cast<const uint16_t*>(ptr);
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   return *reinterpret_cast<const uint32_t*>(ptr);
        default: return 0;
    }
}

std::vector<glm::ivec3> readIndices(const tinygltf::Model& m, int idx) {
    auto& acc = m.accessors[idx]; auto& view = m.bufferViews[acc.bufferView];
    auto& buf = m.buffers[view.buffer];
    const unsigned char* data = buf.data.data() + view.byteOffset + acc.byteOffset;
    size_t es = componentSize(acc.componentType);
    size_t stride = acc.ByteStride(view) ? acc.ByteStride(view) : es;
    std::vector<glm::ivec3> faces; faces.reserve(acc.count/3);
    for (size_t i=0;i+2<acc.count;i+=3){
        faces.emplace_back(readIdx(data+i*stride,acc.componentType),
                           readIdx(data+(i+1)*stride,acc.componentType),
                           readIdx(data+(i+2)*stride,acc.componentType));
    }
    return faces;
}

ImageData decodeImage(const tinygltf::Model& model, const tinygltf::Image& image) {
    ImageData out;
    if (image.bufferView<0) return out;
    auto& view=model.bufferViews[image.bufferView]; auto& buf=model.buffers[view.buffer];
    int w,h,ch;
    unsigned char* decoded=stbi_load_from_memory(buf.data.data()+view.byteOffset,static_cast<int>(view.byteLength),&w,&h,&ch,4);
    if (!decoded) return out;
    out.width=w; out.height=h; out.channels=4;
    out.pixels.assign(decoded,decoded+w*h*4);
    stbi_image_free(decoded);
    return out;
}

SceneData loadGLB(const std::string& path) {
    tinygltf::Model model; tinygltf::TinyGLTF loader;
    std::string err, warn;
    if (!loader.LoadBinaryFromFile(&model,&err,&warn,path)) {
        std::cerr << "GLB load error: " << err << "\n"; return {};
    }
    SceneData scene;
    for (auto& img : model.images) scene.images.push_back(decodeImage(model,img));
    for (auto& mat : model.materials) {
        MaterialData m;
        m.doubleSided = mat.doubleSided;
        if (mat.values.count("baseColorFactor")) {
            auto col=mat.values.at("baseColorFactor").ColorFactor();
            m.baseColorFactor=glm::vec4(col[0],col[1],col[2],col[3]);
        }
        if (mat.pbrMetallicRoughness.baseColorTexture.index>=0) {
            int ti=mat.pbrMetallicRoughness.baseColorTexture.index;
            if (ti<(int)model.textures.size()) m.textureIndex=model.textures[ti].source;
        }
        scene.materials.push_back(m);
    }
    if (scene.materials.empty()) scene.materials.push_back(MaterialData{});

    std::function<void(int,const glm::mat4&)> visit=[&](int ni,const glm::mat4& parent){
        auto& node=model.nodes[ni];
        glm::mat4 world=parent*nodeLocalMatrix(node);
        if (node.mesh>=0) {
            auto& mesh=model.meshes[node.mesh];
            glm::mat3 normalMat=glm::transpose(glm::inverse(glm::mat3(world)));
            for (auto& prim:mesh.primitives) {
                if (!prim.attributes.count("POSITION")||prim.indices<0) continue;
                SubMesh out; out.name=node.name;
                out.materialIndex=prim.material>=0?prim.material:0;
                auto localPos=readVec3(model,prim.attributes.at("POSITION"));
                if (prim.attributes.count("NORMAL")) out.normals=readVec3(model,prim.attributes.at("NORMAL"));
                else out.normals.assign(localPos.size(),glm::vec3(0,1,0));
                if (prim.attributes.count("TEXCOORD_0")) out.uvs=readVec2(model,prim.attributes.at("TEXCOORD_0"));
                else out.uvs.assign(localPos.size(),glm::vec2(0));
                out.faces=readIndices(model,prim.indices);
                out.positions.reserve(localPos.size());
                for (size_t i=0;i<localPos.size();++i) {
                    glm::vec3 p=glm::vec3(world*glm::vec4(localPos[i],1));
                    glm::vec3 n=glm::normalize(normalMat*out.normals[i]);
                    out.positions.push_back(p);
                    out.normals[i]=glm::any(glm::isnan(n))?glm::vec3(0,1,0):n;
                    scene.boundsMin=glm::min(scene.boundsMin,p);
                    scene.boundsMax=glm::max(scene.boundsMax,p);
                }
                scene.submeshes.push_back(std::move(out));
            }
        }
        for (int c:node.children) visit(c,world);
    };
    int si=model.defaultScene>=0?model.defaultScene:0;
    if (!model.scenes.empty()) for (int r:model.scenes[si].nodes) visit(r,glm::mat4(1));
    return scene;
}

// ─── Fur generation ──────────────────────────────────────────────────────────
glm::vec3 faceNormal(const glm::vec3& a,const glm::vec3& b,const glm::vec3& c){
    glm::vec3 n=glm::cross(b-a,c-a);
    if (glm::dot(n,n)<1e-10f) return glm::vec3(0,1,0);
    return glm::normalize(n);
}
glm::vec3 makeTangent(const glm::vec3& n){
    glm::vec3 ref=(std::abs(n.x)<0.9f)?glm::vec3(1,0,0):glm::vec3(0,1,0);
    return glm::normalize(glm::cross(n,ref));
}
glm::vec3 sampleMaterialColor(const MaterialData& mat,const ImageData* img,const glm::vec2& uv){
    glm::vec3 color=glm::vec3(mat.baseColorFactor);
    if (!img||img->pixels.empty()||img->width<=0||img->height<=0) return color;
    float u=uv.x-std::floor(uv.x),v=uv.y-std::floor(uv.y);
    int x=glm::clamp((int)(u*(img->width-1)),0,img->width-1);
    int y=glm::clamp((int)(v*(img->height-1)),0,img->height-1);
    size_t idx=(y*img->width+x)*4;
    return color*glm::vec3(img->pixels[idx]/255.f,img->pixels[idx+1]/255.f,img->pixels[idx+2]/255.f);
}
bool shouldGrowFur(const glm::vec3& c){
    return glm::dot(c,glm::vec3(0.2126f,0.7152f,0.0722f))>=0.22f;
}

std::vector<float> generateFur(const SubMesh& mesh,const MaterialData& material,const ImageData* image,bool isHeadMesh){
    std::mt19937 rng(RANDOM_SEED);
    std::uniform_real_distribution<float> rnd01(0,1),rndSym(-1,1);
    const glm::vec3 up(0,1,0);
    glm::vec3 centroid(0);
    for (auto& p:mesh.positions) centroid+=p;
    centroid/=static_cast<float>(std::max<size_t>(mesh.positions.size(),1));
    std::vector<float> verts;
    verts.reserve(mesh.faces.size()*STRANDS_PER_FACE*SEGMENTS*14);
    auto softStep=[](float x,float e0,float e1){
        float t=glm::clamp((x-e0)/(e1-e0),0.f,1.f); return t*t*(3-2*t);
    };
    for (auto& face:mesh.faces){
        const glm::vec3& a=mesh.positions[face.x],&b=mesh.positions[face.y],&c=mesh.positions[face.z];
        glm::vec3 normal=faceNormal(a,b,c);
        glm::vec3 faceCtr=(a+b+c)/3.f;
        glm::vec3 outward=faceCtr-centroid;
        if (glm::dot(normal,outward)<-0.1f) continue;
        glm::vec3 tang=makeTangent(normal);
        glm::vec3 bitang=glm::normalize(glm::cross(normal,tang));
        glm::vec3 toFace=glm::normalize(outward);
        float dy=faceCtr.y-centroid.y;
        float regionalLengthMult=1.f,regionalGravity=0.04f;
        glm::vec3 grownDir=normal;
        int strandsThisFace=STRANDS_PER_FACE;
        if (isHeadMesh){
            glm::vec2 uvCtr=mesh.uvs.empty()?glm::vec2(0):(mesh.uvs[face.x]+mesh.uvs[face.y]+mesh.uvs[face.z])/3.f;
            glm::vec3 fc=sampleMaterialColor(material,image,uvCtr);
            float lum=glm::dot(fc,glm::vec3(0.2126f,0.7152f,0.0722f));
            if (lum<0.08f) continue;
            float wNeck=softStep(-normal.y,0.2f,0.7f);
            regionalLengthMult=(lum>0.72f)?0.70f:0.50f;
            regionalLengthMult=glm::mix(regionalLengthMult,0.60f,wNeck);
            regionalGravity=glm::mix(0.008f,0.06f,wNeck);
            grownDir=glm::normalize(normal*0.96f+toFace*0.04f);
            strandsThisFace=(lum>0.72f)?15:10;
            strandsThisFace=static_cast<int>(strandsThisFace*glm::mix(1.f,2.f,wNeck));
        } else {
            if (faceCtr.y<centroid.y-1.4f) continue;
            if (faceCtr.y>centroid.y+0.55f) continue;
            float wBack=softStep(dy,0.15f,0.40f);
            float wBelly=softStep(-dy,0.08f,0.28f);
            float wSide=softStep(std::abs(toFace.x),0.3f,0.7f)*(1-wBack*0.6f);
            float wFront=softStep(toFace.z,0.f,0.6f);
            float wFace=softStep(dy,0.2f,0.4f)*softStep(toFace.z,0.4f,0.8f);
            float wHead=softStep(dy,0.02f,0.26f);
            bool isLeg=(faceCtr.y<centroid.y-0.05f)&&(std::abs(normal.y)<0.6f);
            float wLeg=isLeg?softStep(centroid.y-faceCtr.y,0.04f,0.28f):0.f;
            wBelly*=(1-wLeg*0.4f);
            glm::vec3 flowBack=glm::normalize(glm::vec3(0,-0.15f,0.30f));
            glm::vec3 flowBelly=glm::normalize(glm::vec3(0,-0.20f,-0.15f));
            glm::vec3 flowSide=glm::normalize(glm::vec3(0,-0.45f,0.10f));
            glm::vec3 flowFront=glm::normalize(glm::vec3(0,-0.35f,-0.28f));
            glm::vec3 flowLeg=glm::normalize(glm::vec3(toFace.x*0.4f,-0.35f,toFace.z*0.4f));
            glm::vec3 flowRump=glm::normalize(glm::vec3(0,-0.25f,0.22f));
            float blendedW=wBack+wBelly+wSide+wFront+wLeg;
            glm::vec3 blendedFlow=flowBack*wBack+flowBelly*wBelly+flowSide*wSide+flowFront*wFront+flowLeg*wLeg;
            blendedFlow+=flowRump*glm::max(0.f,1.f-blendedW);
            blendedFlow=glm::normalize(blendedFlow);
            regionalLengthMult=1.05f*wBack+1.15f*wBelly+1.f*wSide+1.20f*wFront+0.60f*wLeg+0.95f*glm::max(0.f,1.f-blendedW);
            regionalLengthMult=glm::max(regionalLengthMult,0.50f);
            regionalLengthMult=glm::mix(regionalLengthMult,0.75f,glm::clamp(wHead,0.f,1.f));
            regionalLengthMult=glm::mix(regionalLengthMult,0.30f,glm::clamp(wFace,0.f,1.f));
            glm::vec3 flowOnSurface=blendedFlow-glm::dot(blendedFlow,normal)*normal;
            float flowLen=glm::length(flowOnSurface);
            if (flowLen>0.001f){ flowOnSurface=glm::normalize(flowOnSurface); grownDir=glm::normalize(normal*0.40f+flowOnSurface*0.60f); }
            else grownDir=glm::normalize(normal*0.15f+tang*0.85f);
            regionalGravity=0.55f*wBack+0.15f*wBelly+0.45f*wSide+0.40f*wFront+0.08f*wLeg+0.45f*glm::max(0.f,1.f-blendedW);
            float wSum=wBack+wBelly+wSide+wFront+wLeg+glm::max(0.f,1.f-blendedW);
            if (wSum>0.001f) regionalGravity/=wSum;
            regionalGravity=glm::clamp(regionalGravity*0.3f,0.01f,0.15f);
            regionalGravity=glm::mix(regionalGravity,0.05f,glm::clamp(wHead,0.f,1.f));
            regionalGravity=glm::mix(regionalGravity,0.06f,glm::clamp(wFace,0.f,1.f));
            strandsThisFace=std::max(1,(int)std::lround(STRANDS_PER_FACE*glm::mix(0.70f,0.25f,wHead)));
            strandsThisFace=std::max(0,(int)std::lround((float)strandsThisFace*glm::mix(1.f,0.08f,wFace)));
            if (wFace>0.85f) continue;
        }
        for (int si=0;si<strandsThisFace;++si){
            float r1=rnd01(rng),r2=rnd01(rng);
            if (r1+r2>1.f){r1=1-r1;r2=1-r2;} float r3=1-r1-r2;
            glm::vec3 origin=r1*a+r2*b+r3*c;
            glm::vec2 uv(0);
            if (!mesh.uvs.empty()) uv=r1*mesh.uvs[face.x]+r2*mesh.uvs[face.y]+r3*mesh.uvs[face.z];
            glm::vec3 furColor=sampleMaterialColor(material,image,uv);
            if (!shouldGrowFur(furColor)) continue;
            float length=BASE_LENGTH*regionalLengthMult*(1+rndSym(rng)*LENGTH_VARIANCE);
            float curlPh=rnd01(rng)*2*glm::pi<float>();
            float segLen=length/static_cast<float>(SEGMENTS);
            float clumpAngle=rnd01(rng)*2*glm::pi<float>();
            float clumpStr=0.25f+rnd01(rng)*0.20f;
            float leanT=clumpStr*std::cos(clumpAngle)*0.12f;
            float leanB=clumpStr*std::sin(clumpAngle)*0.12f;
            glm::vec3 pos=origin;
            for (int s=0;s<SEGMENTS;++s){
                float t0=(float)s/SEGMENTS, t1=(float)(s+1)/SEGMENTS;
                glm::vec3 dir=glm::normalize(grownDir+leanT*tang+leanB*bitang);
                float angle=CURL_FREQ*2*glm::pi<float>()*t0+curlPh;
                dir+=CURL_AMOUNT*(std::cos(angle)*tang+std::sin(angle)*bitang);
                dir-=regionalGravity*t0*t0*up;
                dir=glm::normalize(dir);
                glm::vec3 next=pos+dir*segLen;
                verts.insert(verts.end(),{pos.x,pos.y,pos.z,furColor.r,furColor.g,furColor.b,t0});
                verts.insert(verts.end(),{next.x,next.y,next.z,furColor.r,furColor.g,furColor.b,t1});
                pos=next;
            }
        }
    }
    return verts;
}

std::vector<float> flattenMesh(const SubMesh& mesh) {
    std::vector<float> out; out.reserve(mesh.faces.size()*24);
    for (auto& f:mesh.faces){
        glm::vec3 fn=faceNormal(mesh.positions[f.x],mesh.positions[f.y],mesh.positions[f.z]);
        for (int idx:{f.x,f.y,f.z}){
            auto& p=mesh.positions[idx];
            glm::vec3 n=mesh.normals.empty()?fn:mesh.normals[idx];
            glm::vec2 uv=mesh.uvs.empty()?glm::vec2(0):mesh.uvs[idx];
            out.insert(out.end(),{p.x,p.y,p.z,n.x,n.y,n.z,uv.x,uv.y});
        }
    }
    return out;
}

GLuint makeTexture(const ImageData& image) {
    if (image.pixels.empty()) return 0;
    GLuint tex=0; glGenTextures(1,&tex); glBindTexture(GL_TEXTURE_2D,tex);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,image.width,image.height,0,GL_RGBA,GL_UNSIGNED_BYTE,image.pixels.data());
    glGenerateMipmap(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D,0);
    return tex;
}

// ─── Umbrella OBJ loader ──────────────────────────────────────────────────────
std::string trimStr(const std::string& s){
    size_t a=s.find_first_not_of(" \t\r\n"); if (a==std::string::npos) return "";
    size_t b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1);
}
bool isCanopyMat(const std::string& n){ return n=="umbrella_3"||n=="umbrella_4"; }
Vec3s umbrellaColor(const std::string& mat,const Vec3s& p){
    if (p.z<18.f) return {0.18f,0.12f,0.08f};
    if (isCanopyMat(mat)) return {0.92f,0.78f,0.24f};
    if (mat=="umbrella_1") return {0.22f,0.16f,0.10f};
    if (mat=="umbrella_2") return {0.16f,0.10f,0.06f};
    return {0.30f,0.20f,0.12f};
}
bool parseFaceTok(const std::string& tok,int& pi,int& ni){
    pi=ni=0;
    size_t fs=tok.find('/');
    if (fs==std::string::npos){pi=std::stoi(tok);return true;}
    pi=std::stoi(tok.substr(0,fs));
    size_t ss=tok.find('/',fs+1); if (ss==std::string::npos) return true;
    if (ss+1<tok.size()) ni=std::stoi(tok.substr(ss+1));
    return true;
}
bool loadUmbrellaMesh(const std::string& path, UmbrellaMesh& mesh){
    std::ifstream in(path); if (!in) return false;
    std::vector<Vec3s> positions,normals;
    mesh.vertices.clear(); mesh.canopyTriangles.clear();
    mesh.boundsMin=glm::vec3(std::numeric_limits<float>::max());
    mesh.boundsMax=glm::vec3(-std::numeric_limits<float>::max());
    std::string currentMat="umbrella_1",line;
    while (std::getline(in,line)){
        if (line.size()<2) continue;
        if (line.rfind("v ",0)==0){
            std::istringstream ss(line.substr(2)); Vec3s p{}; ss>>p.x>>p.y>>p.z;
            positions.push_back(p);
            mesh.boundsMin=glm::min(mesh.boundsMin,glm::vec3(p.x,p.y,p.z));
            mesh.boundsMax=glm::max(mesh.boundsMax,glm::vec3(p.x,p.y,p.z));
        } else if (line.rfind("vn ",0)==0){
            std::istringstream ss(line.substr(3)); Vec3s n{}; ss>>n.x>>n.y>>n.z; normals.push_back(n);
        } else if (line.rfind("usemtl ",0)==0){
            currentMat=trimStr(line.substr(7));
        } else if (line.rfind("f ",0)==0){
            std::istringstream ss(line.substr(2));
            std::vector<std::pair<int,int>> face; std::string tok;
            while(ss>>tok){int pi,ni;if(parseFaceTok(tok,pi,ni))face.emplace_back(pi-1,ni-1);}
            if (face.size()<3) continue;
            for (size_t i=1;i+1<face.size();++i){
                std::array<std::pair<int,int>,3> tri={face[0],face[i],face[i+1]};
                for (auto& [posIdx,normIdx]:tri){
                    if (posIdx<0||(size_t)posIdx>=positions.size()) continue;
                    Vec3s& p=positions[posIdx];
                    Vec3s col=umbrellaColor(currentMat,p);
                    Vec3s nrm={0,0,1};
                    if (normIdx>=0&&(size_t)normIdx<normals.size()) nrm=normals[normIdx];
                    mesh.vertices.push_back({p.x,p.y,p.z,nrm.x,nrm.y,nrm.z,col.x,col.y,col.z,0,0});
                    if (isCanopyMat(currentMat)) mesh.canopyTriangles.push_back(p);
                }
            }
        }
    }
    return !mesh.vertices.empty();
}

// ─── Snow scene helpers ───────────────────────────────────────────────────────
Particle makeParticle(std::mt19937& rng, bool atTop){
    std::uniform_real_distribution<float> xd(-1.05f,1.05f),yd(-1,1),zd(-0.95f,0.95f);
    std::uniform_real_distribution<float> spd(0.22f,0.62f),drd(0.5f,1.8f),phd(0,6.283f),szd(4,9),tod(0,0.35f);
    Particle p{};
    p.x=xd(rng); p.y=atTop?1.05f+tod(rng):yd(rng); p.z=zd(rng);
    p.speed=spd(rng); p.drift=drd(rng); p.phase=phd(rng); p.size=szd(rng);
    return p;
}

std::vector<FlatVertex> buildBackground(){
    return {
        {-1,-1,0,0.08f,0.11f,0.22f},{1,-1,0,0.08f,0.11f,0.22f},{1,1,0,0.03f,0.05f,0.14f},
        {-1,-1,0,0.08f,0.11f,0.22f},{1,1,0,0.03f,0.05f,0.14f},{-1,1,0,0.03f,0.05f,0.14f},
    };
}
std::vector<FlatVertex> buildGroundBand(){
    return {
        {-1,-1,0,0.80f,0.85f,0.93f},{1,-1,0,0.80f,0.85f,0.93f},{1,-0.42f,0,0.78f,0.83f,0.92f},
        {-1,-1,0,0.80f,0.85f,0.93f},{1,-0.42f,0,0.78f,0.83f,0.92f},{-1,-0.42f,0,0.78f,0.83f,0.92f},
    };
}
std::vector<FlatVertex> buildTopViewFloorFill(){
    return {
        {-1,-1,0,0.80f,0.85f,0.93f},{1,-1,0,0.80f,0.85f,0.93f},{1,1,0,0.80f,0.85f,0.93f},
        {-1,-1,0,0.80f,0.85f,0.93f},{1,1,0,0.80f,0.85f,0.93f},{-1,1,0,0.80f,0.85f,0.93f},
    };
}

std::vector<MeshVertex3> buildFloorMesh(){
    const glm::vec3 fc{0.84f,0.88f,0.96f};
    constexpr float xMn=-1.10f,xMx=1.10f,zMn=-1.05f,zMx=1.05f;
    return {
        {xMn,kFloorTopY,zMn,0,1,0,fc.x,fc.y,fc.z,0,0},{xMx,kFloorTopY,zMn,0,1,0,fc.x,fc.y,fc.z,1,0},{xMx,kFloorTopY,zMx,0,1,0,fc.x,fc.y,fc.z,1,1},
        {xMn,kFloorTopY,zMn,0,1,0,fc.x,fc.y,fc.z,0,0},{xMx,kFloorTopY,zMx,0,1,0,fc.x,fc.y,fc.z,1,1},{xMn,kFloorTopY,zMx,0,1,0,fc.x,fc.y,fc.z,0,1},
        {xMn,kFloorBottomY,zMx,0,0,1,fc.x,fc.y,fc.z,0,0},{xMx,kFloorBottomY,zMx,0,0,1,fc.x,fc.y,fc.z,1,0},{xMx,kFloorTopY,zMx,0,0,1,fc.x,fc.y,fc.z,1,1},
        {xMn,kFloorBottomY,zMx,0,0,1,fc.x,fc.y,fc.z,0,0},{xMx,kFloorTopY,zMx,0,0,1,fc.x,fc.y,fc.z,1,1},{xMn,kFloorTopY,zMx,0,0,1,fc.x,fc.y,fc.z,0,1},
        {xMx,kFloorBottomY,zMn,0,0,-1,fc.x,fc.y,fc.z,0,0},{xMn,kFloorBottomY,zMn,0,0,-1,fc.x,fc.y,fc.z,1,0},{xMn,kFloorTopY,zMn,0,0,-1,fc.x,fc.y,fc.z,1,1},
        {xMx,kFloorBottomY,zMn,0,0,-1,fc.x,fc.y,fc.z,0,0},{xMn,kFloorTopY,zMn,0,0,-1,fc.x,fc.y,fc.z,1,1},{xMx,kFloorTopY,zMn,0,0,-1,fc.x,fc.y,fc.z,0,1},
        {xMn,kFloorBottomY,zMn,-1,0,0,fc.x,fc.y,fc.z,0,0},{xMn,kFloorBottomY,zMx,-1,0,0,fc.x,fc.y,fc.z,1,0},{xMn,kFloorTopY,zMx,-1,0,0,fc.x,fc.y,fc.z,1,1},
        {xMn,kFloorBottomY,zMn,-1,0,0,fc.x,fc.y,fc.z,0,0},{xMn,kFloorTopY,zMx,-1,0,0,fc.x,fc.y,fc.z,1,1},{xMn,kFloorTopY,zMn,-1,0,0,fc.x,fc.y,fc.z,0,1},
        {xMx,kFloorBottomY,zMx,1,0,0,fc.x,fc.y,fc.z,0,0},{xMx,kFloorBottomY,zMn,1,0,0,fc.x,fc.y,fc.z,1,0},{xMx,kFloorTopY,zMn,1,0,0,fc.x,fc.y,fc.z,1,1},
        {xMx,kFloorBottomY,zMx,1,0,0,fc.x,fc.y,fc.z,0,0},{xMx,kFloorTopY,zMn,1,0,0,fc.x,fc.y,fc.z,1,1},{xMx,kFloorTopY,zMx,1,0,0,fc.x,fc.y,fc.z,0,1},
    };
}

BokehParticle makeBokehParticle(std::mt19937& rng, bool atTop){
    std::uniform_real_distribution<float> xd(-1.10f,1.10f),yd(-1.15f,1.15f);
    std::uniform_real_distribution<float> spd(0.03f,0.12f),drd(0.6f,2.0f),phd(0,6.2832f);
    std::uniform_real_distribution<float> szd(96,260),ald(0.07f,0.18f),csd(0,1),twd(0,6.2832f),fld(0,0.9f);
    std::bernoulli_distribution fc(0.42);
    BokehParticle p{};
    p.x=xd(rng); p.y=atTop?1.20f:yd(rng);
    p.speed=spd(rng); p.drift=drd(rng); p.phase=phd(rng);
    p.size=szd(rng); p.alpha=ald(rng); p.colorSeed=csd(rng);
    p.twinkle=twd(rng); p.flicker=fc(rng)?fld(rng):0.f;
    return p;
}

// Build the height-field and silhouette from umbrella canopy triangles
glm::mat4 buildUmbrellaModel(const UmbrellaMesh& mesh){
    glm::vec3 center=(mesh.boundsMin+mesh.boundsMax)*0.5f;
    // Rotations match main.cpp angles converted: Z=-0.2rad, Y=-0.1rad, X=-1.55rad
    glm::mat4 ori=glm::mat4(1);
    ori=glm::rotate(glm::mat4(1),glm::radians(-11.5f),glm::vec3(0,0,1))*ori;
    ori=glm::rotate(glm::mat4(1),glm::radians(-5.7f),glm::vec3(0,1,0))*ori;
    ori=glm::rotate(glm::mat4(1),glm::radians(-88.8f),glm::vec3(1,0,0))*ori;
    ori=glm::translate(glm::mat4(1),-center)*ori;
    // Compute oriented bounds
    glm::vec3 oMin(1e9f),oMax(-1e9f);
    std::array<glm::vec3,8> corners;
    for (int i=0;i<8;++i){
        glm::vec3 c((i&1)?mesh.boundsMax.x:mesh.boundsMin.x,
                    (i&2)?mesh.boundsMax.y:mesh.boundsMin.y,
                    (i&4)?mesh.boundsMax.z:mesh.boundsMin.z);
        glm::vec3 t=glm::vec3(ori*glm::vec4(c,1));
        oMin=glm::min(oMin,t); oMax=glm::max(oMax,t); corners[i]=c;
    }
    float orientedExtentX=oMax.x-oMin.x, orientedExtentY=oMax.y-oMin.y, orientedExtentZ=oMax.z-oMin.z;
    float orientedMaxExtent=std::max({orientedExtentX,orientedExtentY,orientedExtentZ});
    float sc=orientedMaxExtent>0.0001f?(1.3f/orientedMaxExtent):1.f;
    glm::mat4 fitted=glm::scale(glm::mat4(1),glm::vec3(sc))*ori;
    glm::vec3 fMin(1e9f),fMax(-1e9f);
    for (auto& c:corners){glm::vec3 t=glm::vec3(fitted*glm::vec4(c,1));fMin=glm::min(fMin,t);fMax=glm::max(fMax,t);}
    glm::vec3 fCenter=(fMin+fMax)*0.5f;
    float targetCX=-0.05f, targetBotY=kFloorTopY+0.01f, targetCZ=-0.04f;
    return glm::translate(glm::mat4(1),glm::vec3(targetCX-fCenter.x, targetBotY-fMin.y+0.20f, targetCZ-fCenter.z))*fitted;
}

UmbrellaSilhouette buildSilhouette(const std::vector<Vec3s>& tris, const glm::mat4& model){
    UmbrellaSilhouette sil; sil.topY.fill(-10.f);
    if (tris.empty()) return sil;
    std::vector<glm::vec3> tfm; tfm.reserve(tris.size());
    float xMin=1e9f,xMax=-1e9f;
    for (auto& p:tris){
        glm::vec3 t=glm::vec3(model*glm::vec4(p.x,p.y,p.z,1));
        tfm.push_back(t); xMin=std::min(xMin,t.x); xMax=std::max(xMax,t.x);
    }
    if (!(xMax>xMin)) return sil;
    sil.xMin=xMin; sil.xMax=xMax;
    for (size_t s=0;s<kSnowSamples;++s){
        float t=(float)s/(kSnowSamples-1);
        float sx=xMin+(xMax-xMin)*t; float bestY=-10.f;
        for (size_t i=0;i+2<tfm.size();i+=3){
            std::array<glm::vec3,3> tri={tfm[i],tfm[i+1],tfm[i+2]};
            for (int e=0;e<3;++e){
                auto& a=tri[e]; auto& b=tri[(e+1)%3];
                float mn=std::min(a.x,b.x),mx=std::max(a.x,b.x);
                if (sx<mn||sx>mx) continue;
                float dx=b.x-a.x;
                if (std::abs(dx)<0.0001f){bestY=std::max(bestY,std::max(a.y,b.y));continue;}
                float al=(sx-a.x)/dx; if (al<0||al>1) continue;
                bestY=std::max(bestY,a.y+(b.y-a.y)*al);
            }
        }
        sil.topY[s]=bestY;
    }
    for (size_t i=1;i+1<sil.topY.size();++i)
        if (sil.topY[i]<-5.f) sil.topY[i]=std::max(sil.topY[i-1],sil.topY[i+1]);
    sil.valid=true; return sil;
}

bool triHeightAtXZ(const glm::vec3& a,const glm::vec3& b,const glm::vec3& c,float x,float z,float& outY){
    float denom=(b.z-c.z)*(a.x-c.x)+(c.x-b.x)*(a.z-c.z);
    if (std::abs(denom)<1e-6f) return false;
    float w1=((b.z-c.z)*(x-c.x)+(c.x-b.x)*(z-c.z))/denom;
    float w2=((c.z-a.z)*(x-c.x)+(a.x-c.x)*(z-c.z))/denom;
    float w3=1-w1-w2;
    if (w1<-0.001f||w2<-0.001f||w3<-0.001f) return false;
    outY=w1*a.y+w2*b.y+w3*c.y; return true;
}

UmbrellaHeightField buildHeightField(const std::vector<Vec3s>& tris, const glm::mat4& model){
    UmbrellaHeightField f; f.topY.fill(-10.f);
    if (tris.empty()) return f;
    std::vector<glm::vec3> tfm; tfm.reserve(tris.size());
    float xMin=1e9f,xMax=-1e9f,zMin=1e9f,zMax=-1e9f;
    for (auto& p:tris){
        glm::vec3 t=glm::vec3(model*glm::vec4(p.x,p.y,p.z,1));
        tfm.push_back(t); xMin=std::min(xMin,t.x); xMax=std::max(xMax,t.x);
        zMin=std::min(zMin,t.z); zMax=std::max(zMax,t.z);
    }
    if (!(xMax>xMin)||!(zMax>zMin)) return f;
    f.xMin=xMin;f.xMax=xMax;f.zMin=zMin;f.zMax=zMax;
    for (size_t zi=0;zi<UmbrellaHeightField::kSamples;++zi){
        float tz=(float)zi/(UmbrellaHeightField::kSamples-1);
        float sz=zMin+(zMax-zMin)*tz;
        for (size_t xi=0;xi<UmbrellaHeightField::kSamples;++xi){
            float tx=(float)xi/(UmbrellaHeightField::kSamples-1);
            float sx=xMin+(xMax-xMin)*tx;
            float bestY=-10.f;
            for (size_t i=0;i+2<tfm.size();i+=3){
                float y; if (triHeightAtXZ(tfm[i],tfm[i+1],tfm[i+2],sx,sz,y)) bestY=std::max(bestY,y);
            }
            f.topY[zi*UmbrellaHeightField::kSamples+xi]=bestY;
        }
    }
    f.valid=true; return f;
}

float umbrellaTopY(const UmbrellaHeightField& f,float x,float z){
    if (!f.valid||x<f.xMin||x>f.xMax||z<f.zMin||z>f.zMax) return -10.f;
    float nx=(x-f.xMin)/(f.xMax-f.xMin), nz=(z-f.zMin)/(f.zMax-f.zMin);
    float fx=nx*(UmbrellaHeightField::kSamples-1), fz=nz*(UmbrellaHeightField::kSamples-1);
    size_t x0=(size_t)fx,z0=(size_t)fz;
    size_t x1=std::min(x0+1,UmbrellaHeightField::kSamples-1), z1=std::min(z0+1,UmbrellaHeightField::kSamples-1);
    float ax=fx-x0, az=fz-z0;
    float y00=f.topY[z0*UmbrellaHeightField::kSamples+x0], y10=f.topY[z0*UmbrellaHeightField::kSamples+x1];
    float y01=f.topY[z1*UmbrellaHeightField::kSamples+x0], y11=f.topY[z1*UmbrellaHeightField::kSamples+x1];
    return (y00+(y10-y00)*ax)+(y01+(y11-y01)*ax-(y00+(y10-y00)*ax))*az;
}

bool hitsUmbrella(const UmbrellaHeightField& f,float x,float y,float z){
    float top=umbrellaTopY(f,x,z);
    return top>-5.f&&y<=top+0.01f&&y>=top-0.06f;
}

std::vector<FlatVertex> buildSnowCap(const UmbrellaSilhouette& sil,const std::array<float,kSnowSamples>& load){
    std::vector<FlatVertex> v;
    if (!sil.valid) return v;
    v.reserve((kSnowSamples-1)*6);
    float snowCapOffset = -0.10f;  
    for (size_t i=0;i+1<kSnowSamples;++i){
        float t0=(float)i/(kSnowSamples-1), t1=(float)(i+1)/(kSnowSamples-1);
        float x0=sil.xMin+(sil.xMax-sil.xMin)*t0, x1=sil.xMin+(sil.xMax-sil.xMin)*t1;
        float top0=sil.topY[i]+snowCapOffset, top1=sil.topY[i+1]+snowCapOffset;
        if (top0<-5.f||top1<-5.f) continue;
        float sTop0=top0+0.01f+load[i]*0.018f, sTop1=top1+0.01f+load[i+1]*0.018f;
        v.push_back({x0,top0-0.002f,0,0.97f,0.98f,1.f}); v.push_back({x1,top1-0.002f,0,0.97f,0.98f,1.f}); v.push_back({x1,sTop1,0,1,1,1});
        v.push_back({x0,top0-0.002f,0,0.97f,0.98f,1.f}); v.push_back({x1,sTop1,0,1,1,1}); v.push_back({x0,sTop0,0,1,1,1});
    }
    return v;
}

glm::mat4 buildCameraView(CameraMode mode){
    switch(mode){
        case CameraMode::Front:  return glm::mat4(1);
        case CameraMode::Left:   return glm::rotate(glm::mat4(1),glm::radians(90.f),glm::vec3(0,1,0));
        case CameraMode::Right:  return glm::rotate(glm::mat4(1),glm::radians(-90.f),glm::vec3(0,1,0));
        case CameraMode::Back:   return glm::rotate(glm::mat4(1),glm::radians(180.f),glm::vec3(0,1,0));
        case CameraMode::Above:  return glm::rotate(glm::mat4(1),-1.20f,glm::vec3(1,0,0))*glm::translate(glm::mat4(1),glm::vec3(0,-0.10f,0));
        default: return glm::mat4(1);
    }
}

// ─── Shaders ──────────────────────────────────────────────────────────────────
// Flat (background, ground, snow cap)
const char* VS_FLAT = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
uniform mat4 uTransform;
out vec3 vColor;
void main(){ gl_Position=uTransform*vec4(aPos,1.0); vColor=aColor; }
)";
const char* FS_FLAT = R"(#version 330 core
in vec3 vColor; out vec4 FragColor;
void main(){ FragColor=vec4(vColor,1.0); }
)";

// Mesh (umbrella + cat body)
const char* VS_MESH = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aColor;
layout(location=3) in vec2 aTexCoord;
uniform mat4 uModel;
uniform bool uFlattenToScreen;
out vec3 vNormal; out vec3 vColor; out vec2 vTexCoord;
void main(){
    vec4 wp=uModel*vec4(aPos,1.0);
    if (uFlattenToScreen) gl_Position=vec4(wp.xy,0.0,1.0);
    else gl_Position=vec4(wp.xyz,1.0);
    vNormal=normalize(mat3(uModel)*aNormal);
    vColor=aColor; vTexCoord=aTexCoord;
}
)";
const char* FS_MESH = R"(#version 330 core
in vec3 vNormal; in vec3 vColor; in vec2 vTexCoord;
uniform sampler2D uTexture; uniform bool uUseTexture;
out vec4 FragColor;
void main(){
    vec3 N=normalize(vNormal);
    vec3 L=normalize(vec3(-0.35,0.85,0.40));
    float d=abs(dot(N,L));
    float lighting=0.82+0.18*d;
    vec3 base=vColor;
    if (uUseTexture) base*=texture(uTexture,vTexCoord).rgb;
    FragColor=vec4(min(base*lighting,vec3(1.0)),1.0);
}
)";

// Cat body with tinygltf materials
const char* VS_BODY = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
uniform mat4 MVP;
out vec3 vNorm; out vec2 vUV;
void main(){ vNorm=aNormal; vUV=aUV; gl_Position=MVP*vec4(aPos,1.0); }
)";
const char* FS_BODY = R"(#version 330 core
in vec3 vNorm; in vec2 vUV; out vec4 FragColor;
uniform vec4 uBaseColorFactor; uniform sampler2D uBaseColorTex; uniform bool uUseTexture;
void main(){
    vec3 base=uBaseColorFactor.rgb;
    if (uUseTexture) base*=texture(uBaseColorTex,vUV).rgb;
    vec3 N=normalize(vNorm); vec3 L=normalize(vec3(1,2,1.5));
    float d=clamp(dot(N,L),0.0,1.0);
    FragColor=vec4(mix(base*0.55,base,d*0.7+0.3),uBaseColorFactor.a);
}
)";

// Fur
const char* VS_FUR = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
layout(location=2) in float aAlong;
uniform mat4 MVP;
out vec3 vColor; out float vAlong;
void main(){ vColor=aColor; vAlong=aAlong; gl_Position=MVP*vec4(aPos,1.0); }
)";
const char* FS_FUR = R"(#version 330 core
in vec3 vColor; in float vAlong; out vec4 FragColor;
void main(){
    vec3 root=vColor*0.90,mid=vColor*0.98,tip=mix(vColor,vec3(1.0),0.04);
    vec3 col=(vAlong<0.5)?mix(root,mid,vAlong*2.0):mix(mid,tip,(vAlong-0.5)*2.0);
    float alpha=mix(0.55,0.02,pow(vAlong,1.1));
    if (alpha<0.025) discard;
    FragColor=vec4(col,alpha);
}
)";

// Bokeh background lights
const char* VS_BOKEH = R"(#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in float aSize;
layout(location=2) in float aAlpha;
layout(location=3) in float aColorSeed;
layout(location=4) in float aTwinkle;
layout(location=5) in float aFlicker;
out float vAlpha; out float vColorSeed; out float vTwinkle; out float vFlicker;
void main(){
    gl_Position=vec4(aPos,0.0,1.0); gl_PointSize=aSize;
    vAlpha=aAlpha; vColorSeed=aColorSeed; vTwinkle=aTwinkle; vFlicker=aFlicker;
}
)";
const char* FS_BOKEH = R"(#version 330 core
in float vAlpha; in float vColorSeed; in float vTwinkle; in float vFlicker;
uniform float uTime;
out vec4 FragColor;
vec3 cityLightColor(float t){
    vec3 c0=vec3(1.00,0.78,0.52),c1=vec3(1.00,0.93,0.72),c2=vec3(0.72,0.84,1.00),c3=vec3(0.82,0.66,1.00),c4=vec3(0.62,1.00,0.82);
    if (t<0.24) return mix(c0,c1,t/0.24);
    if (t<0.50) return mix(c1,c2,(t-0.24)/0.26);
    if (t<0.74) return mix(c2,c3,(t-0.50)/0.24);
    return mix(c3,c4,(t-0.74)/0.26);
}
void main(){
    vec2 p=gl_PointCoord-vec2(0.5); float d=length(p);
    if (d>0.5) discard;
    float edge=smoothstep(0.52,0.18,d), core=smoothstep(0.26,0.0,d);
    float flickerRate=0.08+0.16*fract(vTwinkle*0.159);
    float flickerPulse=0.76+0.24*sin(uTime*flickerRate+vTwinkle);
    float flickerMix=(1.0-vFlicker)+vFlicker*flickerPulse;
    vec3 color=cityLightColor(vColorSeed)*0.88;
    FragColor=vec4(color,vAlpha*flickerMix*(0.62*edge+0.22*core));
}
)";

// Snow particles
const char* VS_PARTICLE = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in float aSize;
uniform mat4 uView;
void main(){
    vec4 vp=uView*vec4(aPos,1.0);
    gl_Position=vp;
    float ds=clamp(1.15-0.45*vp.z,0.55,1.35);
    gl_PointSize=aSize*ds;
}
)";
const char* FS_PARTICLE = R"(#version 330 core
out vec4 FragColor;
void main(){
    vec2 pt=gl_PointCoord-vec2(0.5);
    float d=length(pt);
    if (d>0.5) discard;
    float alpha=smoothstep(0.5,0.0,d);
    FragColor=vec4(0.95,0.97,1.0,0.95*alpha);
}
)";

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::string glbPath = "assets/im_sol_arm_out.glb";
    if (argc > 1) glbPath = argv[1];

    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(W, H, "Lovely Runner Snow Scene", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cerr << "GLAD init failed\n"; return -1; }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // ── Load cat GLB(s) ───────────────────────────────────────────────────────
    SceneData scene = loadGLB(glbPath);
    if (scene.submeshes.empty()) { std::cerr << "Failed to load GLB\n"; return -1; }

    std::string glbPath2 = "";
    if (argc > 2) glbPath2 = argv[2];
    SceneData scene2;
    bool hasSecondCat = false;
    if (!glbPath2.empty()) {
        scene2 = loadGLB(glbPath2);
        hasSecondCat = !scene2.submeshes.empty();
        if (!hasSecondCat) std::cerr << "Warning: failed to load second GLB: " << glbPath2 << "\n";
    }

    glm::vec3 sceneCenter = 0.5f*(scene.boundsMin+scene.boundsMax);
    glm::vec3 sceneExtent = scene.boundsMax-scene.boundsMin;
    float sceneRadius = 0.5f*glm::length(sceneExtent);
    orbitTarget = sceneCenter+glm::vec3(0,-sceneExtent.y*0.02f,0);
    orbitDist = std::max(0.8f, sceneRadius/std::tan(glm::radians(22.5f))*1.45f);

    std::vector<GLuint> catTextures(scene.images.size(), 0);
    for (size_t i=0;i<scene.images.size();++i) catTextures[i]=makeTexture(scene.images[i]);

    // Build cat GPU meshes (body + fur)
    struct GPUMesh { GLuint vao=0,vbo=0; GLsizei count=0; int matIdx=0; bool hasFur=false; };
    std::vector<GPUMesh> bodyGPU, furGPU;
    for (auto& sub : scene.submeshes) {
        // Body
        GPUMesh body; body.matIdx=sub.materialIndex;
        auto bv=flattenMesh(sub);
        glGenVertexArrays(1,&body.vao); glGenBuffers(1,&body.vbo);
        glBindVertexArray(body.vao); glBindBuffer(GL_ARRAY_BUFFER,body.vbo);
        glBufferData(GL_ARRAY_BUFFER,bv.size()*sizeof(float),bv.data(),GL_STATIC_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)0); glEnableVertexAttribArray(0);
        glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
        glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(6*sizeof(float))); glEnableVertexAttribArray(2);
        body.count=(GLsizei)(bv.size()/8);
        bodyGPU.push_back(body);
        // Fur
        GPUMesh fur; fur.matIdx=sub.materialIndex;
        bool isHead=(sub.name=="Roundcube"), isBody=(sub.name=="Roundcube.001");
        if (isHead||isBody){
            auto& mat=scene.materials[std::clamp(sub.materialIndex,0,(int)scene.materials.size()-1)];
            const ImageData* img=(mat.textureIndex>=0&&mat.textureIndex<(int)scene.images.size())?&scene.images[mat.textureIndex]:nullptr;
            auto fv=generateFur(sub,mat,img,isHead);
            if (!fv.empty()){
                glGenVertexArrays(1,&fur.vao); glGenBuffers(1,&fur.vbo);
                glBindVertexArray(fur.vao); glBindBuffer(GL_ARRAY_BUFFER,fur.vbo);
                glBufferData(GL_ARRAY_BUFFER,fv.size()*sizeof(float),fv.data(),GL_STATIC_DRAW);
                glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,7*sizeof(float),(void*)0); glEnableVertexAttribArray(0);
                glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,7*sizeof(float),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
                glVertexAttribPointer(2,1,GL_FLOAT,GL_FALSE,7*sizeof(float),(void*)(6*sizeof(float))); glEnableVertexAttribArray(2);
                fur.count=(GLsizei)(fv.size()/7); fur.hasFur=true;
            }
        }
        furGPU.push_back(fur);
    }

    float catMaxExtent = std::max({sceneExtent.x, sceneExtent.y, sceneExtent.z});
    float catScale = catMaxExtent>0.0001f?(0.72f/catMaxExtent):1.f;
    float halfH = sceneExtent.y*0.5f*catScale;
    float catY = (kFloorTopY+0.01f)+halfH;
    glm::mat4 catModel = glm::translate(glm::mat4(1),glm::vec3(0.20f, catY - 0.1f, -0.04f))
        * glm::scale(glm::mat4(1),glm::vec3(catScale))
        * glm::rotate(glm::mat4(1), 1.30f,glm::vec3(0,1,0))
        * glm::scale(glm::mat4(1),glm::vec3(1,1,-1))
        * glm::translate(glm::mat4(1),-sceneCenter);

    // ── Load second cat GLB (sunjae) ──────────────────────────────────────────
    std::vector<GLuint> cat2Textures;
    std::vector<GPUMesh> body2GPU, fur2GPU;
    glm::mat4 cat2Model(1);
    if (hasSecondCat) {
        glm::vec3 scene2Center = 0.5f*(scene2.boundsMin+scene2.boundsMax);
        glm::vec3 scene2Extent = scene2.boundsMax-scene2.boundsMin;

        cat2Textures.resize(scene2.images.size(), 0);
        for (size_t i=0;i<scene2.images.size();++i) cat2Textures[i]=makeTexture(scene2.images[i]);

        for (auto& sub : scene2.submeshes) {
            // Body
            GPUMesh body; body.matIdx=sub.materialIndex;
            auto bv=flattenMesh(sub);
            glGenVertexArrays(1,&body.vao); glGenBuffers(1,&body.vbo);
            glBindVertexArray(body.vao); glBindBuffer(GL_ARRAY_BUFFER,body.vbo);
            glBufferData(GL_ARRAY_BUFFER,bv.size()*sizeof(float),bv.data(),GL_STATIC_DRAW);
            glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)0); glEnableVertexAttribArray(0);
            glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
            glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(6*sizeof(float))); glEnableVertexAttribArray(2);
            body.count=(GLsizei)(bv.size()/8);
            body2GPU.push_back(body);
            // Fur
            GPUMesh fur; fur.matIdx=sub.materialIndex;
            bool isHead=(sub.name=="Roundcube"), isBody=(sub.name=="Roundcube.001");
            if (isHead||isBody){
                auto& mat=scene2.materials[std::clamp(sub.materialIndex,0,(int)scene2.materials.size()-1)];
                const ImageData* img=(mat.textureIndex>=0&&mat.textureIndex<(int)scene2.images.size())?&scene2.images[mat.textureIndex]:nullptr;
                auto fv=generateFur(sub,mat,img,isHead);
                if (!fv.empty()){
                    glGenVertexArrays(1,&fur.vao); glGenBuffers(1,&fur.vbo);
                    glBindVertexArray(fur.vao); glBindBuffer(GL_ARRAY_BUFFER,fur.vbo);
                    glBufferData(GL_ARRAY_BUFFER,fv.size()*sizeof(float),fv.data(),GL_STATIC_DRAW);
                    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,7*sizeof(float),(void*)0); glEnableVertexAttribArray(0);
                    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,7*sizeof(float),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
                    glVertexAttribPointer(2,1,GL_FLOAT,GL_FALSE,7*sizeof(float),(void*)(6*sizeof(float))); glEnableVertexAttribArray(2);
                    fur.count=(GLsizei)(fv.size()/7); fur.hasFur=true;
                }
            }
            fur2GPU.push_back(fur);
        }

        float cat2ExtentY = scene2Extent.y;
        float cat2Scale = (std::max({scene2Extent.x, scene2Extent.y, scene2Extent.z}) > 0.0001f)
            ? (1.00f / std::max({scene2Extent.x, scene2Extent.y, scene2Extent.z})) : 1.f;
        float half2H = cat2ExtentY * 0.5f * cat2Scale;
        float cat2Y = (kFloorTopY + 0.01f) + half2H;
        // Offset down so the bottom of the bounding box sits on the floor
        cat2Model = glm::translate(glm::mat4(1), glm::vec3(-0.35f, cat2Y - 0.1f, -0.10f))
            * glm::scale(glm::mat4(1), glm::vec3(cat2Scale))
            * glm::rotate(glm::mat4(1), -1.40f, glm::vec3(0,1,0))
            * glm::scale(glm::mat4(1), glm::vec3(1,1,-1))
            * glm::translate(glm::mat4(1), -scene2Center);
    }

    // ── Load umbrella OBJ ─────────────────────────────────────────────────────
    UmbrellaMesh umbrellaMesh;
    std::vector<std::string> umbPaths={
        "assets/umbrella/12981_umbrella_v1_l2.obj",
        "./assets/umbrella/12981_umbrella_v1_l2.obj",
        "../assets/umbrella/12981_umbrella_v1_l2.obj"
    };
    for (auto& p:umbPaths) if (loadUmbrellaMesh(p,umbrellaMesh)) break;
    if (umbrellaMesh.vertices.empty()) {
        std::cerr << "Warning: umbrella OBJ not found — umbrella will be missing\n";
    }
    glm::mat4 umbrellaModel(1);
    UmbrellaSilhouette silhouette;
    UmbrellaHeightField heightField;
    if (!umbrellaMesh.vertices.empty()){
        umbrellaModel=buildUmbrellaModel(umbrellaMesh);
        silhouette=buildSilhouette(umbrellaMesh.canopyTriangles,umbrellaModel);
        heightField=buildHeightField(umbrellaMesh.canopyTriangles,umbrellaModel);
    }

    // ── Build programs ────────────────────────────────────────────────────────
    GLuint flatProg   = makeProgram(VS_FLAT,    FS_FLAT);
    GLuint meshProg   = makeProgram(VS_MESH,    FS_MESH);
    GLuint bodyProg   = makeProgram(VS_BODY,    FS_BODY);
    GLuint furProg    = makeProgram(VS_FUR,     FS_FUR);
    GLuint bokehProg  = makeProgram(VS_BOKEH,   FS_BOKEH);
    GLuint particleProg = makeProgram(VS_PARTICLE, FS_PARTICLE);

    // ── Background quad VAO ───────────────────────────────────────────────────
    auto background=buildBackground(), groundBand=buildGroundBand();
    auto setupFlatVAO=[](const std::vector<FlatVertex>& v) -> std::pair<GLuint,GLuint> {
        GLuint vao,vbo; glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,v.size()*sizeof(FlatVertex),v.data(),GL_STATIC_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(FlatVertex),(void*)offsetof(FlatVertex,x)); glEnableVertexAttribArray(0);
        glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(FlatVertex),(void*)offsetof(FlatVertex,r)); glEnableVertexAttribArray(1);
        return {vao,vbo};
    };
    auto [bgVao,bgVbo]=setupFlatVAO(background);
    auto [gbVao,gbVbo]=setupFlatVAO(groundBand);
    auto [topFloorVao,topFloorVbo]=setupFlatVAO(buildTopViewFloorFill());

    // ── Floor mesh VAO ────────────────────────────────────────────────────────
    auto floorMesh=buildFloorMesh();
    GLuint floorVao=0,floorVbo=0;
    glGenVertexArrays(1,&floorVao); glGenBuffers(1,&floorVbo);
    glBindVertexArray(floorVao); glBindBuffer(GL_ARRAY_BUFFER,floorVbo);
    glBufferData(GL_ARRAY_BUFFER,floorMesh.size()*sizeof(MeshVertex3),floorMesh.data(),GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(MeshVertex3),(void*)offsetof(MeshVertex3,x)); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(MeshVertex3),(void*)offsetof(MeshVertex3,nx)); glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,sizeof(MeshVertex3),(void*)offsetof(MeshVertex3,r)); glEnableVertexAttribArray(2);
    glVertexAttribPointer(3,2,GL_FLOAT,GL_FALSE,sizeof(MeshVertex3),(void*)offsetof(MeshVertex3,u)); glEnableVertexAttribArray(3);

    // Snow cap VAO (dynamic)
    GLuint snowCapVao,snowCapVbo;
    glGenVertexArrays(1,&snowCapVao); glGenBuffers(1,&snowCapVbo);
    glBindVertexArray(snowCapVao); glBindBuffer(GL_ARRAY_BUFFER,snowCapVbo);
    glBufferData(GL_ARRAY_BUFFER,(kSnowSamples-1)*6*sizeof(FlatVertex),nullptr,GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(FlatVertex),(void*)offsetof(FlatVertex,x)); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(FlatVertex),(void*)offsetof(FlatVertex,r)); glEnableVertexAttribArray(1);

    // ── Umbrella VAO ──────────────────────────────────────────────────────────
    GLuint umbVao=0,umbVbo=0;
    if (!umbrellaMesh.vertices.empty()){
        glGenVertexArrays(1,&umbVao); glGenBuffers(1,&umbVbo);
        glBindVertexArray(umbVao); glBindBuffer(GL_ARRAY_BUFFER,umbVbo);
        glBufferData(GL_ARRAY_BUFFER,umbrellaMesh.vertices.size()*sizeof(MeshVertex3),umbrellaMesh.vertices.data(),GL_STATIC_DRAW);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(MeshVertex3),(void*)offsetof(MeshVertex3,x)); glEnableVertexAttribArray(0);
        glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(MeshVertex3),(void*)offsetof(MeshVertex3,nx)); glEnableVertexAttribArray(1);
        glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,sizeof(MeshVertex3),(void*)offsetof(MeshVertex3,r)); glEnableVertexAttribArray(2);
        glVertexAttribPointer(3,2,GL_FLOAT,GL_FALSE,sizeof(MeshVertex3),(void*)offsetof(MeshVertex3,u)); glEnableVertexAttribArray(3);
    }

    // ── Particle VAOs ─────────────────────────────────────────────────────────
    std::mt19937 rng(std::random_device{}());
    std::vector<Particle> particles; particles.reserve(kParticleCount);
    for (size_t i=0;i<kParticleCount;++i) particles.push_back(makeParticle(rng,false));
    std::vector<ParticleVertex> particleVerts(kParticleCount);

    GLuint partVao,partVbo;
    glGenVertexArrays(1,&partVao); glGenBuffers(1,&partVbo);
    glBindVertexArray(partVao); glBindBuffer(GL_ARRAY_BUFFER,partVbo);
    glBufferData(GL_ARRAY_BUFFER,particleVerts.size()*sizeof(ParticleVertex),nullptr,GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(ParticleVertex),(void*)offsetof(ParticleVertex,x)); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,1,GL_FLOAT,GL_FALSE,sizeof(ParticleVertex),(void*)offsetof(ParticleVertex,size)); glEnableVertexAttribArray(1);

    // Settled snow on umbrella
    std::vector<ParticleVertex> settledVerts(UmbrellaHeightField::kSamples*UmbrellaHeightField::kSamples);
    GLuint settledVao,settledVbo;
    glGenVertexArrays(1,&settledVao); glGenBuffers(1,&settledVbo);
    glBindVertexArray(settledVao); glBindBuffer(GL_ARRAY_BUFFER,settledVbo);
    glBufferData(GL_ARRAY_BUFFER,settledVerts.size()*sizeof(ParticleVertex),nullptr,GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(ParticleVertex),(void*)offsetof(ParticleVertex,x)); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,1,GL_FLOAT,GL_FALSE,sizeof(ParticleVertex),(void*)offsetof(ParticleVertex,size)); glEnableVertexAttribArray(1);

    // ── Bokeh VAO ─────────────────────────────────────────────────────────────
    std::vector<BokehParticle> bokehParticles; bokehParticles.reserve(kBokehCount);
    for (size_t i=0;i<kBokehCount;++i) bokehParticles.push_back(makeBokehParticle(rng,false));
    std::vector<BokehVertex> bokehVerts(kBokehCount);

    GLuint bokehVao=0,bokehVbo=0;
    glGenVertexArrays(1,&bokehVao); glGenBuffers(1,&bokehVbo);
    glBindVertexArray(bokehVao); glBindBuffer(GL_ARRAY_BUFFER,bokehVbo);
    glBufferData(GL_ARRAY_BUFFER,bokehVerts.size()*sizeof(BokehVertex),nullptr,GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(BokehVertex),(void*)offsetof(BokehVertex,x)); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,1,GL_FLOAT,GL_FALSE,sizeof(BokehVertex),(void*)offsetof(BokehVertex,size)); glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,1,GL_FLOAT,GL_FALSE,sizeof(BokehVertex),(void*)offsetof(BokehVertex,alpha)); glEnableVertexAttribArray(2);
    glVertexAttribPointer(3,1,GL_FLOAT,GL_FALSE,sizeof(BokehVertex),(void*)offsetof(BokehVertex,colorSeed)); glEnableVertexAttribArray(3);
    glVertexAttribPointer(4,1,GL_FLOAT,GL_FALSE,sizeof(BokehVertex),(void*)offsetof(BokehVertex,twinkle)); glEnableVertexAttribArray(4);
    glVertexAttribPointer(5,1,GL_FLOAT,GL_FALSE,sizeof(BokehVertex),(void*)offsetof(BokehVertex,flicker)); glEnableVertexAttribArray(5);

    glBindVertexArray(0);

    // Set up uniform locations for body shader
    glUseProgram(bodyProg);
    glUniform1i(glGetUniformLocation(bodyProg,"uBaseColorTex"),0);
    GLint bodyMVPLoc       = glGetUniformLocation(bodyProg,"MVP");
    GLint bodyColorFactorLoc= glGetUniformLocation(bodyProg,"uBaseColorFactor");
    GLint bodyUseTexLoc    = glGetUniformLocation(bodyProg,"uUseTexture");
    GLint furMVPLoc        = glGetUniformLocation(furProg,"MVP");

    std::array<float,kSnowSamples> snowLoad{}; snowLoad.fill(0);
    std::array<float,UmbrellaHeightField::kSamples*UmbrellaHeightField::kSamples> settledLoad{}; settledLoad.fill(0);

    float lastTime = static_cast<float>(glfwGetTime());

    // ── Render loop ───────────────────────────────────────────────────────────
    while (!glfwWindowShouldClose(window)) {
        float now = static_cast<float>(glfwGetTime());
        float dt  = std::min(now-lastTime, 0.033f);
        lastTime  = now;

        if (glfwGetKey(window,GLFW_KEY_ESCAPE)==GLFW_PRESS) glfwSetWindowShouldClose(window,true);
        bool spaceDown=glfwGetKey(window,GLFW_KEY_SPACE)==GLFW_PRESS;
        if (spaceDown&&!spaceWasDown)
            cameraMode=static_cast<CameraMode>((static_cast<int>(cameraMode)+1)%static_cast<int>(CameraMode::Count));
        spaceWasDown=spaceDown;

        // Decay snow loads
        for (auto& s:snowLoad)    s=std::max(0.f,s-dt*0.025f);
        for (auto& s:settledLoad) s=std::max(0.f,s-dt*0.010f);

        // Update particles
        for (size_t i=0;i<particles.size();++i){
            auto& p=particles[i];
            p.y-=p.speed*dt;
            p.x+=std::sin(now*p.drift+p.phase)*0.16f*dt;
            p.z+=std::cos(now*p.drift*0.8f+p.phase)*0.06f*dt;
            bool hit=hitsUmbrella(heightField,p.x,p.y,p.z);
            if (hit){
                if (heightField.valid&&p.x>=heightField.xMin&&p.x<=heightField.xMax&&p.z>=heightField.zMin&&p.z<=heightField.zMax){
                    float nx=(p.x-heightField.xMin)/(heightField.xMax-heightField.xMin);
                    float nz=(p.z-heightField.zMin)/(heightField.zMax-heightField.zMin);
                    size_t xi=std::min((size_t)(glm::clamp(nx,0.f,0.999f)*UmbrellaHeightField::kSamples),UmbrellaHeightField::kSamples-1);
                    size_t zi=std::min((size_t)(glm::clamp(nz,0.f,0.999f)*UmbrellaHeightField::kSamples),UmbrellaHeightField::kSamples-1);
                    settledLoad[zi*UmbrellaHeightField::kSamples+xi]=std::min(1.f,settledLoad[zi*UmbrellaHeightField::kSamples+xi]+0.18f);
                }
                p=makeParticle(rng,true);
            } else if (p.y<-1.1f||std::abs(p.x)>1.15f||std::abs(p.z)>1.15f){
                p=makeParticle(rng,true);
            }
            particleVerts[i]={p.x,p.y,p.z,p.size};
        }

        // Build settled snow on umbrella
        size_t settledCount=0;
        for (size_t zi=0;zi<UmbrellaHeightField::kSamples;++zi){
            float tz=(float)zi/(UmbrellaHeightField::kSamples-1);
            float z=heightField.zMin+(heightField.zMax-heightField.zMin)*tz;
            for (size_t xi=0;xi<UmbrellaHeightField::kSamples;++xi){
                size_t idx=zi*UmbrellaHeightField::kSamples+xi;
                float load=settledLoad[idx]; if (load<0.04f) continue;
                float tx=(float)xi/(UmbrellaHeightField::kSamples-1);
                float x=heightField.xMin+(heightField.xMax-heightField.xMin)*tx;
                float y=heightField.topY[idx]+0.006f+load*0.03f;
                settledVerts[settledCount++]={x,y,z,2.f+load*5.f};
            }
        }

        // Update bokeh vertices
        for (size_t i=0;i<bokehParticles.size();++i){
            auto& b=bokehParticles[i];
            bokehVerts[i]={b.x,b.y,b.size,b.alpha,b.colorSeed,b.twinkle,b.flicker};
        }

        auto snowCap=buildSnowCap(silhouette,snowLoad);

        // Upload dynamic buffers
        glBindBuffer(GL_ARRAY_BUFFER,snowCapVbo);
        glBufferSubData(GL_ARRAY_BUFFER,0,snowCap.size()*sizeof(FlatVertex),snowCap.data());
        glBindBuffer(GL_ARRAY_BUFFER,partVbo);
        glBufferSubData(GL_ARRAY_BUFFER,0,particleVerts.size()*sizeof(ParticleVertex),particleVerts.data());
        glBindBuffer(GL_ARRAY_BUFFER,settledVbo);
        glBufferSubData(GL_ARRAY_BUFFER,0,settledCount*sizeof(ParticleVertex),settledVerts.data());
        glBindBuffer(GL_ARRAY_BUFFER,bokehVbo);
        glBufferSubData(GL_ARRAY_BUFFER,0,bokehVerts.size()*sizeof(BokehVertex),bokehVerts.data());

        int fbW,fbH; glfwGetFramebufferSize(window,&fbW,&fbH);
        (void)fbW; (void)fbH;
        glm::mat4 cameraView=buildCameraView(cameraMode);

        glClearColor(0.03f,0.05f,0.14f,1.f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        // Background (no depth test)
        glDisable(GL_DEPTH_TEST);
        glm::mat4 identity(1);
        glUseProgram(flatProg);
        glUniformMatrix4fv(glGetUniformLocation(flatProg,"uTransform"),1,GL_FALSE,glm::value_ptr(identity));
        glBindVertexArray(bgVao); glDrawArrays(GL_TRIANGLES,0,(GLsizei)background.size());
        if (cameraMode==CameraMode::Above){
            glBindVertexArray(topFloorVao); glDrawArrays(GL_TRIANGLES,0,6);
        } else {
            glBindVertexArray(gbVao); glDrawArrays(GL_TRIANGLES,0,(GLsizei)groundBand.size());
        }

        // Bokeh (background, no depth)
        glUseProgram(bokehProg);
        glUniform1f(glGetUniformLocation(bokehProg,"uTime"),now);
        glBindVertexArray(bokehVao); glDrawArrays(GL_POINTS,0,(GLsizei)bokehVerts.size());

        glEnable(GL_DEPTH_TEST);

        // Floor (3D slab, only in non-Above modes)
        if (cameraMode!=CameraMode::Above){
            glUseProgram(meshProg);
            glUniform1i(glGetUniformLocation(meshProg,"uFlattenToScreen"),0);
            glUniform1i(glGetUniformLocation(meshProg,"uUseTexture"),0);
            glm::mat4 floorMV=cameraView*glm::translate(glm::mat4(1),glm::vec3(0,0.24f,0));
            glUniformMatrix4fv(glGetUniformLocation(meshProg,"uModel"),1,GL_FALSE,glm::value_ptr(floorMV));
            glBindVertexArray(floorVao); glDrawArrays(GL_TRIANGLES,0,(GLsizei)floorMesh.size());
        }

        // Umbrella
        if (umbVao){
            glUseProgram(meshProg);
            glUniform1i(glGetUniformLocation(meshProg,"uFlattenToScreen"),0);
            glUniform1i(glGetUniformLocation(meshProg,"uUseTexture"),0);
            glm::mat4 umbMVP=cameraView*umbrellaModel;
            glUniformMatrix4fv(glGetUniformLocation(meshProg,"uModel"),1,GL_FALSE,glm::value_ptr(umbMVP));
            glBindVertexArray(umbVao);
            glDrawArrays(GL_TRIANGLES,0,(GLsizei)umbrellaMesh.vertices.size());
        } 

        // Cat body (with tinygltf textures)
        glm::mat4 catMVP = cameraView * catModel;
        glUseProgram(bodyProg);
        glUniformMatrix4fv(bodyMVPLoc,1,GL_FALSE,glm::value_ptr(catMVP));
        for (auto& mesh:bodyGPU){
            auto& mat=scene.materials[std::clamp(mesh.matIdx,0,(int)scene.materials.size()-1)];
            bool useTex=mat.textureIndex>=0&&mat.textureIndex<(int)catTextures.size()&&catTextures[mat.textureIndex]!=0;
            glUniform4fv(bodyColorFactorLoc,1,glm::value_ptr(mat.baseColorFactor));
            glUniform1i(bodyUseTexLoc,useTex?1:0);
            if (useTex){ glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,catTextures[mat.textureIndex]); }
            else glBindTexture(GL_TEXTURE_2D,0);
            glBindVertexArray(mesh.vao); glDrawArrays(GL_TRIANGLES,0,mesh.count);
        }

        // Cat fur
        glDepthMask(GL_FALSE);
        glUseProgram(furProg);
        glUniformMatrix4fv(furMVPLoc,1,GL_FALSE,glm::value_ptr(catMVP));
        for (auto& mesh:furGPU){
            if (!mesh.hasFur||mesh.count==0) continue;
            glBindVertexArray(mesh.vao); glDrawArrays(GL_LINES,0,mesh.count);
        }
        glDepthMask(GL_TRUE);

        // Second cat body (sunjae)
        if (hasSecondCat) {
            glm::mat4 cat2MVP = cameraView * cat2Model;
            glUseProgram(bodyProg);
            glUniformMatrix4fv(bodyMVPLoc,1,GL_FALSE,glm::value_ptr(cat2MVP));
            for (auto& mesh:body2GPU){
                auto& mat=scene2.materials[std::clamp(mesh.matIdx,0,(int)scene2.materials.size()-1)];
                bool useTex=mat.textureIndex>=0&&mat.textureIndex<(int)cat2Textures.size()&&cat2Textures[mat.textureIndex]!=0;
                glUniform4fv(bodyColorFactorLoc,1,glm::value_ptr(mat.baseColorFactor));
                glUniform1i(bodyUseTexLoc,useTex?1:0);
                if (useTex){ glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,cat2Textures[mat.textureIndex]); }
                else glBindTexture(GL_TEXTURE_2D,0);
                glBindVertexArray(mesh.vao); glDrawArrays(GL_TRIANGLES,0,mesh.count);
            }
            // Second cat fur
            glDepthMask(GL_FALSE);
            glUseProgram(furProg);
            glUniformMatrix4fv(furMVPLoc,1,GL_FALSE,glm::value_ptr(cat2MVP));
            for (auto& mesh:fur2GPU){
                if (!mesh.hasFur||mesh.count==0) continue;
                glBindVertexArray(mesh.vao); glDrawArrays(GL_LINES,0,mesh.count);
            }
            glDepthMask(GL_TRUE);
        }

        // Snow particles
        glUseProgram(particleProg);
        glUniformMatrix4fv(glGetUniformLocation(particleProg,"uView"),1,GL_FALSE,glm::value_ptr(cameraView));
        glBindVertexArray(settledVao); glDrawArrays(GL_POINTS,0,(GLsizei)settledCount);
        glBindVertexArray(partVao);    glDrawArrays(GL_POINTS,0,(GLsizei)particleVerts.size());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    for (auto& m:bodyGPU){ glDeleteVertexArrays(1,&m.vao); glDeleteBuffers(1,&m.vbo); }
    for (auto& m:furGPU){ if(m.vao){glDeleteVertexArrays(1,&m.vao);glDeleteBuffers(1,&m.vbo);} }
    for (auto t:catTextures) if(t) glDeleteTextures(1,&t);
    for (auto& m:body2GPU){ glDeleteVertexArrays(1,&m.vao); glDeleteBuffers(1,&m.vbo); }
    for (auto& m:fur2GPU){ if(m.vao){glDeleteVertexArrays(1,&m.vao);glDeleteBuffers(1,&m.vbo);} }
    for (auto t:cat2Textures) if(t) glDeleteTextures(1,&t);
    glDeleteVertexArrays(1,&bgVao); glDeleteBuffers(1,&bgVbo);
    glDeleteVertexArrays(1,&gbVao); glDeleteBuffers(1,&gbVbo);
    glDeleteVertexArrays(1,&topFloorVao); glDeleteBuffers(1,&topFloorVbo);
    glDeleteVertexArrays(1,&floorVao); glDeleteBuffers(1,&floorVbo);
    glDeleteVertexArrays(1,&bokehVao); glDeleteBuffers(1,&bokehVbo);
    glDeleteVertexArrays(1,&snowCapVao); glDeleteBuffers(1,&snowCapVbo);
    if (umbVao){ glDeleteVertexArrays(1,&umbVao); glDeleteBuffers(1,&umbVbo); }
    glDeleteVertexArrays(1,&partVao); glDeleteBuffers(1,&partVbo);
    glDeleteVertexArrays(1,&settledVao); glDeleteBuffers(1,&settledVbo);
    glDeleteProgram(flatProg); glDeleteProgram(meshProg);
    glDeleteProgram(bodyProg); glDeleteProgram(furProg);
    glDeleteProgram(bokehProg); glDeleteProgram(particleProg);
    glfwTerminate();
    return 0;
}