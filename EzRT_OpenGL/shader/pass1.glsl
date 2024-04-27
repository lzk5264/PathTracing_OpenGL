#version 330 core

in vec3 pixPos;
out vec4 FragColor;

uniform samplerBuffer triangles;
uniform samplerBuffer BVHNodes;

uniform int trianglesNum;
uniform int nodesNum;
uniform int width;
uniform int height;
uniform sampler2D hdrMap;

uniform int frameCounter;
uniform sampler2D lastFrame;

const float INF = 1e20;
const float PI = 3.1415926;
const float PDF_Hemi = 1.0 / (2.0 * PI);
const int SIZE_TRIANGLES = 12;
const int SIZE_BVHNODE = 4;
const int MAX_DEPTH = 8;

vec2 SampleSphericalMap(vec3 v) {
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv /= vec2(2.0 * PI, PI);
    uv += 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

vec3 sampleHdr(vec3 v) {
    vec2 uv = SampleSphericalMap(normalize(v));
    vec3 color = texture2D(hdrMap, uv).rgb;
    //color = min(color, vec3(10));
    return color;
}

uint seed = uint(
    uint((pixPos.x * 0.5 + 0.5) * width)  * uint(1973) + 
    uint((pixPos.y * 0.5 + 0.5) * height) * uint(9277) + 
    uint(frameCounter) * uint(26699)) | uint(1);

uint WangHash(inout uint seed) {
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}

float Random()
{
	return float(WangHash(seed)) / 4294967296.0;
}

vec3 SampleHemiSphere() {
    float z = Random();
    float r = max(0, sqrt(1.0 - z*z));
    float phi = 2.0 * PI * Random();
    return vec3(r * cos(phi), r * sin(phi), z);
}

vec3 ConvertToNormalHemi(vec3 v, vec3 N)
{
    vec3 helper = vec3(1, 0, 0);
    if(abs(N.x)>0.999) helper = vec3(0, 0, 1);
    vec3 tangent = normalize(cross(N, helper));
    vec3 bitangent = normalize(cross(N, tangent));
    return v.x * tangent + v.y * bitangent + v.z * N;
}

struct Triangle
{
	vec3 p0, p1, p2;
	vec3 n0, n1, n2;
};

struct Material
{
	vec3 emissive;
	vec3 baseColor;
	float subsurface;
	float metallic;
	float specular;
	float specularTint;
	float roughness;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
	float IOR;
	float transmission;
};


struct BVHNode
{
	int left, right;
	int n, idx;
	vec3 AA, BB;
};

struct Ray
{
	vec3 origin;
	vec3 direction;
};

struct HitResult
{
	bool isHit;
	bool isInside;	//Whether the hitpoint is inside the object;
	float distance; //the distance from hitpoint to ray.origin;
	vec3 pos;		//the position of HitPoint
	vec3 normal;
	vec3 viewDir;	//the direction of ray hitting the point
	Material material;
};

Triangle getTriangle(int i)// get the i th triangle
{
	int offset = i * SIZE_TRIANGLES;
	Triangle t;
	t.p0 = texelFetch(triangles, offset + 0).xyz;
	t.p1 = texelFetch(triangles, offset + 1).xyz;
	t.p2 = texelFetch(triangles, offset + 2).xyz;
	t.n0 = texelFetch(triangles, offset + 3).xyz;
	t.n1 = texelFetch(triangles, offset + 4).xyz;
	t.n2 = texelFetch(triangles, offset + 5).xyz;

	return t;
}

Material getMaterial(int i)
{
	Material m;
	int offset = i * SIZE_TRIANGLES;
	m.emissive = texelFetch(triangles, offset + 6).xyz;
	m.baseColor = texelFetch(triangles, offset + 7).xyz;

	vec3 param1 = texelFetch(triangles, offset + 8).xyz;
	vec3 param2 = texelFetch(triangles, offset + 9).xyz;
	vec3 param3 = texelFetch(triangles, offset + 10).xyz;
	vec3 param4 = texelFetch(triangles, offset + 11).xyz;

	m.subsurface = param1.x;
	m.metallic = param1.y;
	m.specular = param1.z;
	m.specularTint = param2.x;
	m.roughness = param2.y;
	m.anisotropic = param2.z;
	m.sheen = param3.x;
	m.sheenTint = param3.y;
	m.clearcoat = param3.z;
	m.clearcoatGloss = param4.x;
	m.IOR = param4.y;
	m.transmission = param4.z;

	return m;
}

/*
* 3 / 29 / 2024	ZEEKANG
* Get the element with index i in BVH array
*/
BVHNode getBVHNode(int i)
{
	BVHNode node;
	int offset = i * SIZE_BVHNODE;
	vec3 param1 = texelFetch(BVHNodes, offset + 0).xyz;
	vec3 param2 = texelFetch(BVHNodes, offset + 1).xyz;
	node.left = int(param1.x);
	node.right = int(param1.y);
	node.n = int(param2.x);
	node.idx = int(param2.y);
	node.AA = texelFetch(BVHNodes, offset + 2).xyz;
	node.BB = texelFetch(BVHNodes, offset + 3).xyz;
	return node;
}

HitResult IntersectWithTriangle(Triangle triangle, Ray ray)
{
    HitResult res;
    res.isHit = false;
    res.isInside = false;
    res.distance = INF;

    vec3 p0 = triangle.p0;
    vec3 p1 = triangle.p1;
    vec3 p2 = triangle.p2;
    vec3 O = ray.origin;
    vec3 D = ray.direction;

    // 计算三角形的法向量
    vec3 N = normalize(cross(p1 - p0, p2 - p0));
    float NdotRayDirection = dot(D, N);

    // 判断光线是否与三角形的法向量同方向（光线从背面入射）
    if (NdotRayDirection > 0.0f)
    {
        N = -N; // 如果是，反转法线方向
        res.isInside = true; // 并标记为从内部相交
    }

    // 如果光线与法向量垂直，则没有交点
    if (abs(NdotRayDirection) < 0.00001f) return res;

    vec3 E1 = p1 - p0;
    vec3 E2 = p2 - p0;
    vec3 S = O - p0;
    vec3 S1 = cross(D, E2);
    vec3 S2 = cross(S, E1);
    float divisor = dot(S1, E1);
    
    if (divisor == 0.0f) return res; // 如果除数为0，则光线与三角形平行，无交点
    
    float invDivisor = 1.0 / divisor;
    float b1 = dot(S1, S) * invDivisor;
    float b2 = dot(S2, D) * invDivisor;
    float t = dot(S2, E2) * invDivisor;

    // 检查交点是否在三角形内部
    if (b1 < 0.0f || b1 > 1.0f || b2 < 0.0f || (b1 + b2) > 1.0f || t < 1e-9) return res;

    res.isHit = true;
    res.distance = t;
    res.pos = O + t * D;

    // 计算平滑法线
    vec3 Nsmooth = normalize((1 - b1 - b2) * triangle.n0 + b1 * triangle.n1 + b2 * triangle.n2);
    // 应用之前确定的方向
    res.normal = (res.isInside) ? -Nsmooth : Nsmooth;

    return res;
}


HitResult IntersectWithAllTriangle(Ray ray, int l, int r)
{
	HitResult res;
	res.isHit = false;
	res.distance = INF;
	for (int i = l; i <= r; i ++ )
	{
		Triangle triangle = getTriangle(i);
		HitResult r = IntersectWithTriangle(triangle, ray);
		if (r.isHit && r.distance < res.distance)
		{
			res = r;
			res.material = getMaterial(i);
		}
	}

	return res;
}

bool IntersectWithAABB(Ray ray, vec3 AA, vec3 BB)
{
	vec3 inD = 1.0 / (ray.direction + 1e-9);
	vec3 t0s = (AA  - ray.origin) * inD;
	vec3 t1s = (BB - ray.origin) * inD;
	// Find the time to enter and exit the AABB box on three axis.
	vec3 tNear = min(t0s, t1s);
	vec3 tFar = max(t0s, t1s);
	
	float tEnter = max(tNear.x, max(tNear.y, tNear.z));
	float tExit = min(tFar.x, min(tFar.y, tFar.z));

	return tExit >= max(tEnter, 0.0);
}

HitResult IntersectWithBVH(Ray ray)
{
	int stack[256];
	int spointer = 0;

	HitResult res;
	res.isHit = false;
	res.distance = INF;

	stack[spointer++] = 0;
	while (spointer > 0)
	{
		int top = stack[--spointer];
		BVHNode node = getBVHNode(top);
		if (IntersectWithAABB(ray, node.AA, node.BB))
		{
			if (node.n > 0)
			{
				HitResult tmpRes = IntersectWithAllTriangle(ray, node.idx, node.idx + node.n - 1);
				if (tmpRes.isHit && tmpRes.distance < res.distance) res = tmpRes;
				continue;
			}
			else
			{
				if (node.left != -1)
				{
					stack[spointer++] = node.left;
				}
				if (node.right != -1)
				{
					stack[spointer++] = node.right;
				}
			}
		}
		
	}

	return res;
}

float SchlickFresnel(float u)
{
	float m = clamp(1 - u, 0, 1);
	float m2 = m * m;
	return m2 * m2 * m;
}

vec3 BRDF_Evaluate(vec3 V, vec3 N,vec3 L, in Material material)
{
	float NdotL = dot(N, L);
	float NdotV = dot(N, V);
	if (NdotL < 0 || NdotV < 0) return vec3(0);

	vec3 H = normalize(V + L);
	float NdotH = dot(N, H);
	float LdotH = dot(L, H);

	vec3 Cdlin = material.baseColor;

	float Fd90 = 0.5 + 2.0 * LdotH * LdotH * material.roughness;
	float FL = SchlickFresnel(NdotL);
	float FV = SchlickFresnel(NdotV);
	float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

	float Fss90 = LdotH * LdotH * material.roughness;
	float Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
	float ss = 1.25 * (Fss * (1.0 / (NdotL + NdotV) - 0.5) + 0.5);

	vec3 diffuse = (1.0 / PI) * mix(Fd, ss, material.subsurface) * Cdlin;;
	return diffuse * (1.0 - material.metallic);

}

vec3 PathTracing(HitResult res, int MAX_DEPTH)
{
	vec3 Lo = vec3(0);
	vec3 history = vec3(1);

	for (int bounce = 0; bounce < MAX_DEPTH; bounce++ )
	{
		vec3 V = -res.viewDir;
		vec3 N = res.normal;
		vec3 L = ConvertToNormalHemi(SampleHemiSphere(), res.normal);
		float cosine = max(0, dot(res.normal, L));

		Ray randomRay;
		randomRay.origin = res.pos;
		randomRay.direction = L;
		HitResult newRes = IntersectWithBVH(randomRay);


		vec3 f_r = res.material.baseColor / PI;
		if (!newRes.isHit)
		{
			vec3 skyColor = sampleHdr(randomRay.direction);
			Lo += history * skyColor * f_r * cosine / PDF_Hemi;
			break;
		}

		vec3 Le = newRes.material.emissive;
		Lo += history * Le * cosine * f_r / PDF_Hemi;

		res = newRes;

		history *= cosine * f_r / PDF_Hemi;
	}
		return Lo;
}


void main()
{
	Ray ray;
	ray.origin = vec3(0, 0, 3);
	ray.direction = normalize(vec3(pixPos.xy, 2) - ray.origin);
	vec3 Color = vec3(0.0, 0.0, 0.0);
	HitResult firstHit = IntersectWithBVH(ray);
	if (firstHit.isHit)
	{		
		vec3 Le = firstHit.material.emissive;
		vec3 Li = PathTracing(firstHit, MAX_DEPTH);
		Color = Le + Li;
	}
	else
	{
		Color = sampleHdr(ray.direction);
	}

	vec3 lastColor = texture2D(lastFrame, pixPos.xy * 0.5 + 0.5).rgb;

	Color = mix(lastColor, Color, 1.0 / float(frameCounter + 1));

	gl_FragData[0] = vec4(Color , 1.0);
}