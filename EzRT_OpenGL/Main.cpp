#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Headfile/Shader.h"
#include "Headfile/Camera.h"
#define STB_IMAGE_IMPLEMENTATION
#include "Headfile/model.h"
#include "Headfile/hdrloader.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <numbers>
#include <iostream>


const unsigned int SCR_WIDTH = 1024;
const unsigned int SCR_HEIGHT = 1024;
//记录每一帧的鼠标位置
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
//是否第一次聚焦程序
bool firstMouse = true;
//记录每一帧的时间
float deltaTime = 0.0f;
float lastFrame = 0.0f;
int frameCounter = 0;

Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));



void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}


void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        camera.ProcessKeyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        camera.ProcessKeyboard(DOWN, deltaTime);
}

void mouse_callback(GLFWwindow* window, double xPosIn, double yPosIn)
{
    float xPos = static_cast<float>(xPosIn), yPos = static_cast<float>(yPosIn);
    float xOffset, yOffset;
    if (firstMouse)
    {
        lastX = xPos;
        lastY = yPos;
        firstMouse = false;
    }
    xOffset = xPos - lastX;
    yOffset = lastY - yPos;
    camera.ProcessMouseMovement(xOffset, yOffset);
    lastX = xPos;
    lastY = yPos;
}

void scoll_callback(GLFWwindow* window, double xOffset, double yOffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yOffset));
}


//Disney PBR Material
struct Material {
    glm::vec3 emissive = glm::vec3(0.0f, 0.0f, 0.0f);  // 作为光源时的发光颜色
    glm::vec3 baseColor = glm::vec3(0.0f, 0.0f, 0.0f);
    float subsurface = 0.0f;
    float metallic = 0.0f;
    float specular = 0.0f;
    float specularTint = 0.0f;
    float roughness = 0.0f;
    float anisotropic = 0.0f;
    float sheen = 0.0f;
    float sheenTint = 0.0f;
    float clearcoat = 0.0f;
    float clearcoatGloss = 0.0f;
    float IOR = 1.0f;
    float transmission = 0.0f;
};

struct Triangle {
    glm::vec3 p0, p1, p2;    // 顶点坐标
    glm::vec3 n0, n1, n2;    // 顶点法线
    
    std::shared_ptr<Material> material;  // 材质
};

void readObjToTriangles(const std::string& filepath, std::vector<Triangle>& triangles, std::shared_ptr<Material> material, glm::mat4 trans, bool smoothNormal) {
    //vertex position, index
    std::vector<glm::vec3> vertices;
    std::vector<unsigned int> indices;

    std::ifstream fin(filepath);
    std::string line;
    if (!fin.is_open()) {
        std::cout << "ERROR: " << filepath << " OPEN FAILED" << std::endl;
        exit(-1);
    }

    glm::vec3 max(std::numeric_limits<float>::min());
    glm::vec3 min(std::numeric_limits<float>::max());
    // 按行读取
    while (std::getline(fin, line)) {
        std::istringstream sin(line);   // 以一行的数据作为 string stream 解析并且读取
        std::string type;
        GLfloat x, y, z;
        int v0, v1, v2;
        int vn0, vn1, vn2;
        int vt0, vt1, vt2;
        char slash;

        // 统计斜杆数目，用不同格式读取
        int slashCnt = 0;
        for (int i = 0; i < line.length(); i++) {
            if (line[i] == '/') slashCnt++;
        }

        // 读取obj文件
        sin >> type;
        if (type == "v") {
            sin >> x >> y >> z;
            glm::vec3 tmp(x, y, z);
            vertices.push_back(tmp);
            max = glm::max(max, tmp);
            min = glm::min(min, tmp);
        }
        if (type == "f") {
            if (slashCnt == 6) {
                sin >> v0 >> slash >> vt0 >> slash >> vn0;
                sin >> v1 >> slash >> vt1 >> slash >> vn1;
                sin >> v2 >> slash >> vt2 >> slash >> vn2;
            }
            else if (slashCnt == 3) {
                sin >> v0 >> slash >> vt0;
                sin >> v1 >> slash >> vt1;
                sin >> v2 >> slash >> vt2;
            }
            else {
                sin >> v0 >> v1 >> v2;
            }
            indices.push_back(v0 - 1);
            indices.push_back(v1 - 1);
            indices.push_back(v2 - 1);
        }
    }

    glm::vec3 extent(max - min);
    float maxaxis = std::max(extent.x, std::max(extent.y, extent.z));


    for (auto& v : vertices) {
        v.x /= maxaxis;
        v.y /= maxaxis;
        v.z /= maxaxis;
    }

    // 通过矩阵进行坐标变换
    for (auto& v : vertices) {
        glm::vec4 vv = glm::vec4(v.x, v.y, v.z, 1);
        vv = trans * vv;
        v = glm::vec3(vv.x, vv.y, vv.z);
    }

    // 生成法线
    std::vector<glm::vec3> normals(vertices.size(), glm::vec3(0, 0, 0));
    for (int i = 0; i < indices.size(); i += 3) {
        glm::vec3 p0 = vertices[indices[i]];
        glm::vec3 p1 = vertices[indices[i + 1]];
        glm::vec3 p2 = vertices[indices[i + 2]];
        glm::vec3 n = normalize(cross(p1 - p0, p2 - p0));
        normals[indices[i]] += n;
        normals[indices[i + 1]] += n;
        normals[indices[i + 2]] += n;
    }

    // 构建 Triangle 对象数组
    int offset = triangles.size();  // 增量更新
    triangles.resize(offset + indices.size() / 3);
    for (int i = 0; i < indices.size(); i += 3) {
        Triangle& t = triangles[offset + i / 3];
        // 传顶点属性
        t.p0 = vertices[indices[i]];
        t.p1 = vertices[indices[i + 1]];
        t.p2 = vertices[indices[i + 2]];
        if (!smoothNormal) {
            glm::vec3 n = glm::normalize(glm::cross(t.p1 - t.p0, t.p2 - t.p0));
            t.n0 = n; t.n1 = n; t.n2 = n;
        }
        else {
            t.n0 = glm::normalize(normals[indices[i]]);
            t.n1 = glm::normalize(normals[indices[i + 1]]);
            t.n2 = glm::normalize(normals[indices[i + 2]]);
        }

        // 传材质
        t.material = material;
    }
}

struct TriangleEncoded
{
    glm::vec3 p0, p1, p2;   // 顶点坐标
    glm::vec3 n0, n1, n2;   // 顶点法线
    // 材质
    glm::vec3 emissive;
    glm::vec3 baseColor;
    glm::vec3 param1;   // subsurface, metallic, specular
    glm::vec3 param2;   // specularTint, roughness, anisotropic
    glm::vec3 param3;   // sheen, sheenTint, clearcoat
    glm::vec3 param4;   // clearcoatGloss, IOR, transmission
};

struct BVHNode
{
    int left = -1, right = -1;    // child tree
    int n = 0, idx;     // if n != 0, this node is a left node.
                        // n is the num of triangles contained in this leaf node
                        // idx is the index of the first triangle contained in this leaf node in the Triangle array
    glm::vec3 AA, BB;   //AABB
};


int BuildBVH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n)
{
    if (l > r) return 0;
    
    nodes.emplace_back(BVHNode());
    int idx(nodes.size() - 1);
    // calculate AABB
    glm::vec3 min(std::numeric_limits<float>::max());
    glm::vec3 max(std::numeric_limits<float>::min());
    for (int i = l; i <= r; i++)
    {
        min = glm::min(min, glm::min(triangles[i].p0, glm::min(triangles[i].p1, triangles[i].p2)));
        max = glm::max(max, glm::max(triangles[i].p0, glm::max(triangles[i].p1, triangles[i].p2)));
    }
    nodes[idx].AA = min;
    nodes[idx].BB = max;
    // Satisfy the condition for leaf node
    if (r - l + 1 <= n)
    {
        nodes[idx].n = r - l + 1;
        nodes[idx].idx = l;
        return idx;
    }
    // select split axis
    glm::vec3 extent(max - min);
    int axis((extent.x > extent.y) ? ((extent.x > extent.z) ? 0 : 2) : ((extent.y > extent.z) ? 1 : 2));
    // select split point
    std::sort(triangles.begin() + l, triangles.begin() + r + 1, 
        [axis](const Triangle& t1, const Triangle& t2) 
        {
            glm::vec3 t1Center = (t1.p0 + t1.p1 + t1.p2) / 3.0f;
            glm::vec3 t2Center = (t2.p0 + t2.p1 + t2.p2) / 3.0f;
            return t1Center[axis] < t2Center[axis];
        });
    int mid((l + r) >> 1);
    //recursive tree building
    nodes[idx].left = BuildBVH(triangles, nodes, l, mid, n);
    nodes[idx].right = BuildBVH(triangles, nodes, mid + 1, r, n);


    return idx;
}


int BuildBVHWithSAH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n)
{
    if (l > r) return 0;

    nodes.emplace_back(BVHNode());
    int idx(nodes.size() - 1);
    // calculate AABB
    glm::vec3 min(std::numeric_limits<float>::max());
    glm::vec3 max(std::numeric_limits<float>::min());
    for (int i = l; i <= r; i++)
    {
        min = glm::min(min, glm::min(triangles[i].p0, glm::min(triangles[i].p1, triangles[i].p2)));
        max = glm::max(max, glm::max(triangles[i].p0, glm::max(triangles[i].p1, triangles[i].p2)));
    }
    nodes[idx].AA = min;
    nodes[idx].BB = max;
    // Satisfy the condition for leaf node
    if (r - l + 1 <= n)
    {
        nodes[idx].n = r - l + 1;
        nodes[idx].idx = l;
        return idx;
    }
    // SAH: find better split way

    glm::vec3 parentExtent(max - min);
    float parentS(2.0f * (parentExtent.x * parentExtent.y + parentExtent.y * parentExtent.z + parentExtent.z * parentExtent.x));
    
    float bestCost(std::numeric_limits<float>::max());
    int bestAxis, bestSplit;

    for (int axis = 0; axis < 3; axis++)
    {
        std::sort(triangles.begin() + l, triangles.begin() + r + 1,
            [axis](const Triangle& t1, const Triangle& t2) {return (t1.p0[axis] + t1.p1[axis] + t1.p2[axis]) < (t2.p0[axis] + t2.p1[axis] + t2.p2[axis]); });
        std::vector<glm::vec3> leftAA(r - l + 1, glm::vec3(std::numeric_limits<float>::max()));
        std::vector<glm::vec3> leftBB(r - l + 1, glm::vec3(std::numeric_limits<float>::min()));
        std::vector<glm::vec3> rightAA(r - l + 1, glm::vec3(std::numeric_limits<float>::max()));
        std::vector<glm::vec3> rightBB(r - l + 1, glm::vec3(std::numeric_limits<float>::min()));

        for (int i = l; i <= r; i++)
        {
            int bias((i == l) ? 0 : 1);
            leftAA[i - l] = glm::min(leftAA[i - l - bias], glm::min(triangles[i].p0, glm::min(triangles[i].p1, triangles[i].p2)));
            leftBB[i - l] = glm::max(leftBB[i - l - bias], glm::max(triangles[i].p0, glm::max(triangles[i].p1, triangles[i].p2)));
        }

        for (int i = r; i >= l; i--)
        {
            int bias((i == r) ? 0 : 1);
            rightAA[i - l] = glm::min(rightAA[i - l + bias], glm::min(triangles[i].p0, glm::min(triangles[i].p1, triangles[i].p2)));
            rightBB[i - l] = glm::max(rightBB[i - l + bias], glm::max(triangles[i].p0, glm::max(triangles[i].p1, triangles[i].p2)));
        }

        for (int split = l; split <= r; split++)
        {
            glm::vec3 leftExtent(leftBB[split - l] - leftAA[split - l]);
            glm::vec3 rightExtent(rightBB[split - l] - rightAA[split - l]);
            float leftS(2.0f * (leftExtent.x * leftExtent.y + leftExtent.y * leftExtent.z + leftExtent.z * leftExtent.x));
            float rightS(2.0f * (rightExtent.x * rightExtent.y + rightExtent.y * rightExtent.z + rightExtent.z * rightExtent.x));

            float cost((leftS * (split - l + 1) + rightS * (r - split)) / parentS);

            if (cost < bestCost)
            {
                bestCost = cost;
                bestAxis = axis;
                bestSplit = split;
            }
        }
    }

    std::sort(triangles.begin() + l, triangles.begin() + r + 1,
        [bestAxis](const Triangle& t1, const Triangle& t2) { return (t1.p0[bestAxis] + t1.p1[bestAxis] + t1.p2[bestAxis]) < (t2.p0[bestAxis] + t2.p1[bestAxis] + t2.p2[bestAxis]); });

    nodes[idx].left = BuildBVHWithSAH(triangles, nodes, l, bestSplit, n);
    nodes[idx].right = BuildBVHWithSAH(triangles, nodes, bestSplit + 1, r, n);

    return idx;
}

struct BVHEncoded
{
    glm::vec3 param1;   // left, right, null
    glm::vec3 param2;   // n, idx, null;
    glm::vec3 AA;
    glm::vec3 BB;
};


void ConvertToTriangleEncoded(const std::vector<Triangle>& triangles, std::vector<TriangleEncoded>& trianglesEncoded)
{
    for (int i = 0; i < triangles.size(); i++)
    {
        TriangleEncoded tmp;
        tmp.n0 = triangles[i].n0;
        tmp.n1 = triangles[i].n1;
        tmp.n2 = triangles[i].n2;

        tmp.p0 = triangles[i].p0;
        tmp.p1 = triangles[i].p1;
        tmp.p2 = triangles[i].p2;

        tmp.emissive = triangles[i].material->emissive;
        tmp.baseColor = triangles[i].material->baseColor;

        tmp.param1 = glm::vec3(triangles[i].material->subsurface, triangles[i].material->metallic, triangles[i].material->specular);
        tmp.param2 = glm::vec3(triangles[i].material->specularTint, triangles[i].material->roughness, triangles[i].material->anisotropic);
        tmp.param3 = glm::vec3(triangles[i].material->sheen, triangles[i].material->sheenTint, triangles[i].material->clearcoat);
        tmp.param4 = glm::vec3(triangles[i].material->clearcoatGloss, triangles[i].material->IOR, triangles[i].material->transmission);

        trianglesEncoded.push_back(tmp);
    }
}

void ConvertToBVHEncoded(std::vector<BVHNode>& nodes, std::vector<BVHEncoded>& nodesEncoded)
{
    for (int i = 0; i < nodes.size(); i++)
    {
        BVHEncoded tmp;
        tmp.param1 = glm::vec3(nodes[i].left, nodes[i].right, 0.0f);
        tmp.param2 = glm::vec3(nodes[i].n, nodes[i].idx, 0.0f);
        tmp.AA = nodes[i].AA;
        tmp.BB = nodes[i].BB;
        nodesEncoded.push_back(tmp);
    }
}

unsigned int GenTextureRGB32F(int width, int height)
{
    unsigned int tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}


class RenderPass
{
public:
    unsigned int FBO = 0;
    unsigned int squareVAO, squareVBO;
    std::vector<unsigned int> colorAttachments;
    Shader shaderProgram;
    int width = 1024;
    int height = 1024;

    RenderPass(const char* vertexPath, const char* fragmentPath)
    {
        shaderProgram = Shader(vertexPath, fragmentPath);
    }

    void BindData(bool isFinalPass = false)
    {
        if (!isFinalPass) glGenFramebuffers(1, &FBO);
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);

        glGenBuffers(1, &squareVBO);
        glBindBuffer(GL_ARRAY_BUFFER, squareVBO);

        std::vector<glm::vec3> square = {
            glm::vec3(-1, -1, 0), glm::vec3(1, -1, 0), glm::vec3(-1, 1, 0),
            glm::vec3(1, 1, 0), glm::vec3(-1, 1, 0), glm::vec3(1, -1, 0)
                    };
        glBufferData(GL_ARRAY_BUFFER, square.size() * sizeof(glm::vec3), square.data(), GL_STATIC_DRAW);


        glGenVertexArrays(1, &squareVAO);
        glBindVertexArray(squareVAO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);


        if (!isFinalPass) {
            std::vector<GLuint> attachments;
            for (int i = 0; i < colorAttachments.size(); i++) {
                glBindTexture(GL_TEXTURE_2D, colorAttachments[i]);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorAttachments[i], 0);// 将颜色纹理绑定到 i 号颜色附件
                attachments.push_back(GL_COLOR_ATTACHMENT0 + i);
            }
            glDrawBuffers(attachments.size(), &attachments[0]);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Draw(std::vector<unsigned int> texPassArray = {}) {
        shaderProgram.use();
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);
        glBindVertexArray(squareVAO);
        // 传上一帧的帧缓冲颜色附件
        for (int i = 0; i < texPassArray.size(); i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, texPassArray[i]);
            std::string uName = "texPass" + std::to_string(i);
            shaderProgram.setInt(uName.c_str(), i);

        }
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glBindVertexArray(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glUseProgram(0);
    }
};

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    //--------------------------------------------------------------------------

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scoll_callback);


    //------------------------------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //-----------------------------------------------------------------------------------


    std::vector<Triangle> triangles;

    const std::string bunnyModelPath("Rsrc/Models/Bunny/bunny.obj");
    auto materialBunny = std::make_shared<Material>();
    materialBunny->baseColor = glm::vec3(0, 1, 1);
    glm::mat4 bunnyModel(1.0f);
    bunnyModel = glm::translate(bunnyModel, glm::vec3(0.3, -1.4, 0.1));
    bunnyModel = glm::scale(bunnyModel, glm::vec3(1.5f, 1.5f, 1.5f));

    readObjToTriangles(bunnyModelPath, triangles, materialBunny, bunnyModel, true);

    const std::string sphereModelPath("Rsrc/Models/Sphere/sphere.obj");
    auto materialDiffuse1 = std::make_shared<Material>();
    materialDiffuse1->baseColor = glm::vec3(0.725f, 0.71f, 1.0f);
    materialDiffuse1->roughness = 0.1f;
    materialDiffuse1->metallic = 0.1f;
    materialDiffuse1->clearcoat = 1.0f;
    materialDiffuse1->clearcoatGloss = 0.05;
    materialDiffuse1->subsurface = 1.0f;
    materialDiffuse1->specular = 1.0f;
    materialDiffuse1->specularTint = 0.0f;
    glm::mat4 sphereModelTrans(1.0f);
    //sphereModelTrans = glm::scale(sphereModelTrans, glm::vec3(2.0f, 2.0f, 2.0f));
    //readObjToTriangles(sphereModelPath, triangles, materialDiffuse1, sphereModelTrans, true);


    const std::string quadModelPath("Rsrc/Models/Quad/quad.obj");
    auto materialQuad = std::make_shared<Material>();
    materialQuad->baseColor = glm::vec3(1);
    materialQuad->emissive = glm::vec3(4);
    glm::mat4 quadModel(1.0f);
    quadModel = glm::translate(quadModel, glm::vec3(0.0, 1.2, -0.0));
    quadModel = glm::rotate(quadModel, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    //readObjToTriangles(quadModelPath, triangles, materialQuad, quadModel, false);

    auto materialQuad1 = std::make_shared<Material>();
    materialQuad1->baseColor = glm::vec3(0.725, 0.71, 0.68);
    quadModel = glm::mat4(1.0f);
    quadModel = glm::translate(quadModel, glm::vec3(0.0, -1.38, -0.0));
    quadModel = glm::rotate(quadModel, glm::radians(-30.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    quadModel = glm::scale(quadModel, glm::vec3(2.8f, 2.8f, 2.8));
    //readObjToTriangles(quadModelPath, triangles, materialQuad1, quadModel, false);



    std::vector<BVHNode> nodes;
    BuildBVHWithSAH(triangles, nodes, 0, triangles.size() - 1, 8);
    //std::cout << bunnyModelPath << " BVH building complete, " << nodes.size() << " nodes\n";
    
    std::vector<BVHEncoded> nodesEncoded;
    ConvertToBVHEncoded(nodes, nodesEncoded);

    std::vector<TriangleEncoded> trianglesEncoded;
    ConvertToTriangleEncoded(triangles, trianglesEncoded);
    //---------------------------------------------------------------
    //import HDR Environment Map
    // 图像数据
    HDRLoaderResult hdrImage;

    // 加载HDR图像
    bool loaded = HDRLoader::load("Rsrc/HDREnvironmentMap/stuttgart_suburbs_4k.hdr", hdrImage);
    if (!loaded) {
        std::cout << "HDR importing error" << '\n';
    }
    unsigned int hdrTexture;

    hdrTexture = GenTextureRGB32F(hdrImage.width, hdrImage.height);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, hdrImage.width, hdrImage.height, 0, GL_RGB, GL_FLOAT, hdrImage.cols);

    //-------------------------------------------------------------------------------
   

    // TriangleEncoded TBO
    unsigned int trianglesEncodedTBO;
    glGenBuffers(1, &trianglesEncodedTBO);

    glBindBuffer(GL_TEXTURE_BUFFER, trianglesEncodedTBO);
    glBufferData(GL_TEXTURE_BUFFER, trianglesEncoded.size() * sizeof(TriangleEncoded), trianglesEncoded.data(), GL_STATIC_DRAW);

    unsigned int triEncTex;
    glGenTextures(1, &triEncTex);
    glBindTexture(GL_TEXTURE_BUFFER, triEncTex);

    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, trianglesEncodedTBO);
    
    //  BVNEncoded TBO;
    unsigned int nodesEncodedTBO;
    glGenBuffers(1, &nodesEncodedTBO);

    glBindBuffer(GL_TEXTURE_BUFFER, nodesEncodedTBO);
    glBufferData(GL_TEXTURE_BUFFER, nodesEncoded.size() * sizeof(BVHEncoded), nodesEncoded.data(), GL_STATIC_DRAW);

    unsigned int bvhEncTex;
    glGenTextures(1, &bvhEncTex);
    glBindTexture(GL_TEXTURE_BUFFER, bvhEncTex);

    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, nodesEncodedTBO);

    //  lastFrame Tex
    unsigned int lastFrameTex;


    //-------------------------------------------------------------------------------
    RenderPass pass1("shader/shadervs.glsl", "shader/pass1.glsl");
    //将渲染结果保存到我们新创建的纹理中，而不是渲染到屏幕上
    pass1.colorAttachments.push_back(GenTextureRGB32F(pass1.width, pass1.height));
    pass1.colorAttachments.push_back(GenTextureRGB32F(pass1.width, pass1.height));
    pass1.colorAttachments.push_back(GenTextureRGB32F(pass1.width, pass1.height));
    pass1.BindData();

    pass1.shaderProgram.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, triEncTex);
    pass1.shaderProgram.setInt("triangles", 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, bvhEncTex);
    pass1.shaderProgram.setInt("BVHNodes", 1);

    pass1.shaderProgram.setInt("trianglesNum", trianglesEncoded.size());
    pass1.shaderProgram.setInt("nodesNum", nodes.size());


    pass1.shaderProgram.setInt("width", SCR_WIDTH);
    pass1.shaderProgram.setInt("height", SCR_HEIGHT);
    glUseProgram(0);

    RenderPass pass2("shader/shadervs.glsl", "shader/pass2.glsl");
    lastFrameTex = GenTextureRGB32F(pass2.width, pass2.height);
    pass2.colorAttachments.push_back(lastFrameTex);
    pass2.BindData();

    RenderPass pass3("shader/shadervs.glsl", "shader/pass3.glsl");
    pass3.BindData(true);

    //Rendering Loop
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        std::cout << "\r\33[2KFPS: " << 1 / deltaTime << std::flush;

        // input
        processInput(window);
        glClearColor(0.0, 0.0, 0.0, 1.0);
        pass1.shaderProgram.use();
        pass1.shaderProgram.setInt("frameCounter", frameCounter++);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, hdrTexture);
        pass1.shaderProgram.setInt("hdrMap", 2);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, lastFrameTex);
        pass1.shaderProgram.setInt("lastFrame", 3);


        // render
        pass1.Draw();
        pass2.Draw(pass1.colorAttachments);
        pass3.Draw(pass2.colorAttachments);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glfwTerminate();

    return 0;
}