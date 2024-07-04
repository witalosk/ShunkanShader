
// Permuted Congruential Generator
// Original source code: https://www.shadertoy.com/view/XlGcRh

#define FLOAT_MAX float(0xffffffffu)
#define FLOAT_MAX_INVERT 1.0 / FLOAT_MAX
#define PI 3.1415926535897932384626433832795
#define QUATERNION_IDENTITY float4(0, 0, 0, 1)


uint Pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint2 Pcg2d(uint2 v)
{
    v = v * 1664525u + 1013904223u;

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v = v ^ (v>>16u);

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v = v ^ (v>>16u);

    return v;
}

uint3 Pcg3d(uint3 v) {

    v = v * 1664525u + 1013904223u;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    v ^= v >> 16u;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    return v;
}

uint4 Pcg4d(uint4 v)
{
    v = v * 1664525u + 1013904223u;

    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;

    v ^= v >> 16u;

    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;

    return v;
}

float Pcg01(uint v)
{
    return Pcg(v) * FLOAT_MAX_INVERT;
}

float2 Pcg2d01(uint2 v)
{
    return Pcg2d(v) * FLOAT_MAX_INVERT;
}

float3 Pcg3d01(uint3 v)
{
    return Pcg3d(v) * FLOAT_MAX_INVERT;
}

float4 Pcg4d01(uint4 v)
{
    return Pcg4d(v) * FLOAT_MAX_INVERT;
}



static const float3x3 RgbToYiq = { 0.299, 0.587, 0.114, 0.595716, -0.274453, -0.321263, 0.211456, -0.522591, 0.311135 };
static const float3x3 YiqToRgb = { 1.0, 0.9563, 0.6210, 1.0, -0.2721, -0.6474, 1.0, -1.1070, 1.7046 };
static const float3x3 LinToLms = { 3.90405e-1, 5.49941e-1, 8.92632e-3, 7.08416e-2, 9.63172e-1, 1.35775e-3, 2.31082e-2, 1.28021e-1, 9.36245e-1 };
static const float3x3 LmsToLin = { 2.85847e+0, -1.62879e+0, -2.48910e-2, -2.10182e-1, 1.15820e+0, 3.24281e-4, -4.18120e-2, -1.18169e-1, 1.06867e+0 };

inline half3 AdjustContrast(half3 color, float contrast)
{
    return color < 0.5 ? pow(color * 2, contrast) * 0.5 : (1.0 - pow(2.0 * (1.0 - color), contrast) * 0.5);
}

inline float AdjustContrast(float val, float contrast)
{
    return val < 0.5 ? pow(val * 2, contrast) * 0.5 : (1.0 - pow(2.0 * (1.0 - val), contrast) * 0.5);
}

float3 ConvertRgbToHsv(float3 c)
{
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));

    const float d = q.x - min(q.w, q.y);
    const float e = 1.0e-5;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

float3 ConvertHsvToRgb(float3 c)
{
    const float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    const float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

/**
 * \brief Linear->Gamma変換
 * https://en.wikipedia.org/wiki/SRGB
 */
float3 ConvertLinearToGamma(float3 c)
{
    const float3 linearRGB = saturate(c);
    return saturate(linearRGB <= 0.0031308 ? linearRGB * 12.92 : 1.055 * pow(linearRGB, 0.416666667) - 0.055);
}

/**
 * \brief Gamma->Linear変換
 * https://en.wikipedia.org/wiki/SRGB
 */
float3 ConvertGammaToLinear(float3 c)
{
    const float3 gammaRGB = saturate(c);
    return saturate(gammaRGB <= 0.04045 ? gammaRGB / 12.92 : pow((gammaRGB + 0.055) / 1.055, 2.4));
}

/**
 * \brief HSVをずらす
 */
float3 HsvShift(float3 color, float3 hsvShift)
{
    float3 hsv = ConvertRgbToHsv(color);
    hsv.x += hsvShift.x;
    hsv.x %= 1.0;
    hsv.y = hsv.y * hsvShift.y;
    hsv.z = hsv.z * hsvShift.z;
    return ConvertHsvToRgb(hsv);
}

/**
 * \brief YIQ色空間で色相をずらす
 * https://mofu-dev.com/blog/introducing-yiq/
 */
float3 HueShiftYiq(float3 color, float hueShift)
{
    float3 yColor = mul(RgbToYiq, saturate(color - 0.0000001));

    const float originalHue = atan2(yColor.b, yColor.g);
    const float finalHue = originalHue - hueShift * 2.0 * PI;

    const float chroma = sqrt(yColor.b * yColor.b + yColor.g * yColor.g);
    const float3 yFinalColor = float3(yColor.r, chroma * cos(finalHue), chroma * sin(finalHue));

    return mul(YiqToRgb, yFinalColor);
}

/**
 * \brief ホワイトバランスを調整する
 * \param c 入力色
 * \param balance ホワイトバランス (ColorUtility.ComputeColorBalance()でC#側で計算した値)
 * \return ホワイトバランス調整後の色
 */
float3 WhiteBalance(float3 c, float3 balance)
{
    float3 lms = mul(LinToLms, c);
    lms *= balance;
    return mul(LmsToLin, lms);
}

/**
 * \brief 値の強さによって色相を変える
 */
float3 CalcStrengthColor(float val)
{
    float len = length(val);
    return ConvertHsvToRgb(float3(1.0 - saturate(len), saturate(2.0 - clamp(len, 0.0, 1.25)), len));
}

/**
 * \brief 色の明るさを計算する
 */
inline float CalcLuminance(float3 color)
{
    return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
}


static const int BlendNone = 0;
static const int BlendAlphaBlend = 10;
static const int BlendAdd = 20;
static const int BlendAddLighter = 21;
static const int BlendSubtract = 30;
static const int BlendDifference = 40;
static const int BlendMultiply = 50;
static const int BlendScreen = 60;
static const int BlendOverlay = 70;
static const int BlendDarken = 80;
static const int BlendLighten = 90;
static const int BlendColorDodge = 100;
static const int BlendColorBurn = 110;
static const int BlendHardLight = 120;
static const int BlendSoftLight = 130;


/**
 * \brief Blend Color
 * \param a sourceColor
 * \param b inputColor
 * \param mode blend mode
 * \return blended color
 */
half4 Blend(half4 a, half4 b, int mode)
{
    half4 result = a;
    switch (mode)
    {
        case BlendAlphaBlend:
            result = lerp(a, b, b.a);
            break;
        case BlendAdd:
            result = a + b;
            break;
        case BlendAddLighter:
            result = lerp(a, a + b, pow(CalcLuminance(a.rgb), 0.5));
            break;
        case BlendSubtract:
            result = a - b;
            break;
        case BlendMultiply:
            result = a * b;
            break;
        case BlendScreen:
            result = 1.0 - (1.0 - a) * (1.0 - b);
            break;
        case BlendOverlay:
            result = b < 0.5 ? 2.0 * a * b : 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
            break;
        case BlendDarken:
            result = min(a, b);
            break;
        case BlendLighten:
            result = max(a, b);
            break;
        case BlendColorDodge:
            result = b == 1.0 ? b : min(1.0, a / (1.0 - b));
            break;
        case BlendColorBurn:
            result = b == 0.0 ? b : max(0.0, 1.0 - (1.0 - a) / b);
            break;
        case BlendHardLight:
            result = a < 0.5 ? 2.0 * a * b : 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
            break;
        case BlendSoftLight:
            result = b < 0.5 ? 2.0 * a * b + a * a * (1.0 - 2.0 * b) : sqrt(a) * (2.0 * b - 1.0) + 2.0 * a * (1.0 - b);
            break;
        case BlendDifference:
            result = abs(a - b);
            break;
        default:
            break;
    }

    return result;
}

/**
 * \brief Blend Color
 * \param a sourceColor
 * \param b inputColor
 * \param mode blend mode
 * \return blended color
 */
half3 Blend(half3 a, half3 b, int mode)
{
    half3 result = a;
    switch (mode)
    {
        case BlendAlphaBlend:
            break;
        case BlendAdd:
            result = a + b;
            break;
        case BlendSubtract:
            result = a - b;
            break;
        case BlendMultiply:
            result = a * b;
            break;
        case BlendScreen:
            result = 1.0 - (1.0 - a) * (1.0 - b);
            break;
        case BlendOverlay:
            result = b < 0.5 ? 2.0 * a * b : 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
            break;
        case BlendDarken:
            result = min(a, b);
            break;
        case BlendLighten:
            result = max(a, b);
            break;
        case BlendColorDodge:
            result = b == 1.0 ? b : min(1.0, a / (1.0 - b));
            break;
        case BlendColorBurn:
            result = b == 0.0 ? b : max(0.0, 1.0 - (1.0 - a) / b);
            break;
        case BlendHardLight:
            result = a < 0.5 ? 2.0 * a * b : 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
            break;
        case BlendSoftLight:
            result = b < 0.5 ? 2.0 * a * b + a * a * (1.0 - 2.0 * b) : sqrt(a) * (2.0 * b - 1.0) + 2.0 * a * (1.0 - b);
            break;
        case BlendDifference:
            result = abs(a - b);
            break;
        default:
            break;
    }

    return result;
}


inline float4 Mod289(float4 x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

inline float3 Mod289(float3 x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

inline float2 Mod289(float2 x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

inline float Mod289(float x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

inline float4 Permute(float4 x)
{
    return Mod289(((x * 34.0) + 1.0) * x);
}

inline float3 Permute(float3 x)
{
    return Mod289(((x * 34.0) + 1.0) * x);
}

inline float Permute(float x)
{
    return Mod289(((x * 34.0) + 1.0) * x);
}

float4 TaylorInvSqrt(float4 r)
{
    return rsqrt(r);
}

float TaylorInvSqrt(float r)
{
    return rsqrt(r);
}

float4 Grad4(float j, float4 ip)
{
    const float4 ones = float4(1.0, 1.0, 1.0, -1.0);
    float4 p;

    p.xyz = floor(frac(j.xxx * ip.xyz) * 7.0) * ip.z - 1.0;
    p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
    float4 s = float4(1.0 - step(0.0, p));
    p.xyz = p.xyz + (s.xyz * 2.0 - 1.0) * s.www;

    return p;
}

/**
 * \brief 2D Simplex Noise
 * \param v Input Coordinate
 * \return [-1.0, 1.0]
 */
float SimplexNoise(float2 v)
{
    const float4 C = float4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
        -0.577350269189626,  // -1.0 + 2.0 * C.x
        0.024390243902439); // 1.0 / 41.0
    // First corner
    float2 i = floor(v + dot(v, C.yy));
    float2 x0 = v - i + dot(i, C.xx);

    // Other corners
    float2 i1;
    //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
    //i1.y = 1.0 - i1.x;
    i1 = (x0.x > x0.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);
    // x0 = x0 - 0.0 + 0.0 * C.xx ;
    // x1 = x0 - i1 + 1.0 * C.xx ;
    // x2 = x0 - 1.0 + 2.0 * C.xx ;
    float4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;

    // Permutations
    i = Mod289(i); // Avoid truncation effects in permutation
    float3 p = Permute(Permute(i.y + float3(0.0, i1.y, 1.0))
        + i.x + float3(0.0, i1.x, 1.0));

    float3 m = max(0.5 - float3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;

    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

    float3 x = 2.0 * frac(p * C.www) - 1.0;
    float3 h = abs(x) - 0.5;
    float3 ox = floor(x + 0.5);
    float3 a0 = x - ox;

    // Normalise gradients implicitly by scaling m
    // Approximation of: m *= inversesqrt( a0*a0 + h*h );
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

    // Compute final noise value at P
    float3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

/**
 * \brief 3D Simplex Noise
 * \param v Input Coordinate
 * \return [-1.0, 1.0]
 */
float SimplexNoise(float3 v)
{
    // 定数
    const float2 C = float2(1.0 / 6.0, 1.0 / 3.0);
    const float4 D = float4(0.0, 0.5, 1.0, 2.0);

    float3 i = floor(v + dot(v, C.yyy)); // 変形した座標の整数部
    float3 x0 = v - i + dot(i, C.xxx);  // 単体1つめの頂点 

    float3 g = step(x0.yzx, x0.xyz);	  // 成分比較
    float3 l = 1.0 - g;
    float3 i1 = min(g.xyz, l.zxy);
    float3 i2 = max(g.xyz, l.zxy);

    //     x0 = x0 - 0. + 0.0 * C       // 単体1つめの頂点 
    float3 x1 = x0 - i1 + 1.0 * C.xxx;	// 単体2つめの頂点 
    float3 x2 = x0 - i2 + 2.0 * C.xxx;	// 単体3つめの頂点 
    float3 x3 = x0 - 1. + 3.0 * C.xxx;	// 単体4つめの頂点 

    // 勾配ベクトル計算時のインデックスを並べ替え
    i = Mod289(i);
    float4 p = Permute(Permute(Permute(
                i.z + float4(0.0, i1.z, i2.z, 1.0))
            + i.y + float4(0.0, i1.y, i2.y, 1.0))
        + i.x + float4(0.0, i1.x, i2.x, 1.0));

    // 勾配ベクトルを計算
    float n_ = 0.142857142857; // 1.0 / 7.0
    float3 ns = n_ * D.wyz - D.xzx;

    float4 j = p - 49.0 * floor(p * ns.z * ns.z);	// fmod(p, 7*7)

    float4 x_ = floor(j * ns.z);
    float4 y_ = floor(j - 7.0 * x_); // fmod(j, N)

    float4 x = x_ * ns.x + ns.yyyy;
    float4 y = y_ * ns.x + ns.yyyy;
    float4 h = 1.0 - abs(x) - abs(y);

    float4 b0 = float4(x.xy, y.xy);
    float4 b1 = float4(x.zw, y.zw);

    float4 s0 = floor(b0) * 2.0 + 1.0;
    float4 s1 = floor(b1) * 2.0 + 1.0;
    float4 sh = -step(h, float4(0.0, 0.0, 0.0, 0.0));

    float4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    float4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    float3 p0 = float3(a0.xy, h.x);
    float3 p1 = float3(a0.zw, h.y);
    float3 p2 = float3(a1.xy, h.z);
    float3 p3 = float3(a1.zw, h.w);

    // 勾配を正規化
    float4 norm = TaylorInvSqrt(float4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // 放射円状ブレンドカーネル（放射円状に減衰）
    float4 m = max(0.6 - float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    // 最終的なノイズの値を算出
    return 42.0 * dot(m * m, float4(dot(p0, x0), dot(p1, x1),
        dot(p2, x2), dot(p3, x3)));
}

/**
 * \brief 4D Simplex Noise
 * \param v Input Coordinate
 * \return [-1.0, 1.0]
 */
float SimplexNoise(float4 v)
{
    const float4 C = float4(0.138196601125011,  // (5 - sqrt(5))/20  G4
        0.276393202250021,  // 2 * G4
        0.414589803375032,  // 3 * G4
        -0.447213595499958); // -1 + 4 * G4

    // First corner
    float4 i = floor(v + dot(v, 0.309016994374947451));
    float4 x0 = v - i + dot(i, C.xxxx);

    // Other corners

    // Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
    float4 i0;
    float3 isX = step(x0.yzw, x0.xxx);
    float3 isYZ = step(x0.zww, x0.yyz);
    //  i0.x = dot( isX, float3( 1.0 ) );
    i0.x = isX.x + isX.y + isX.z;
    i0.yzw = 1.0 - isX;
    //  i0.y += dot( isYZ.xy, float2( 1.0 ) );
    i0.y += isYZ.x + isYZ.y;
    i0.zw += 1.0 - isYZ.xy;
    i0.z += isYZ.z;
    i0.w += 1.0 - isYZ.z;

    // i0 now contains the unique values 0,1,2,3 in each channel
    float4 i3 = clamp(i0, 0.0, 1.0);
    float4 i2 = clamp(i0 - 1.0, 0.0, 1.0);
    float4 i1 = clamp(i0 - 2.0, 0.0, 1.0);

    //  x0 = x0 - 0.0 + 0.0 * C.xxxx
    //  x1 = x0 - i1  + 1.0 * C.xxxx
    //  x2 = x0 - i2  + 2.0 * C.xxxx
    //  x3 = x0 - i3  + 3.0 * C.xxxx
    //  x4 = x0 - 1.0 + 4.0 * C.xxxx
    float4 x1 = x0 - i1 + C.xxxx;
    float4 x2 = x0 - i2 + C.yyyy;
    float4 x3 = x0 - i3 + C.zzzz;
    float4 x4 = x0 + C.wwww;

    // Permutations
    i = Mod289(i);
    float j0 = Permute(Permute(Permute(Permute(i.w) + i.z) + i.y) + i.x);
    float4 j1 = Permute(Permute(Permute(Permute(
                    i.w + float4(i1.w, i2.w, i3.w, 1.0))
                + i.z + float4(i1.z, i2.z, i3.z, 1.0))
            + i.y + float4(i1.y, i2.y, i3.y, 1.0))
        + i.x + float4(i1.x, i2.x, i3.x, 1.0));

    // Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
    // 7*7*6 = 294, which is close to the ring size 17*17 = 289.
    float4 ip = float4(1.0 / 294.0, 1.0 / 49.0, 1.0 / 7.0, 0.0);

    float4 p0 = Grad4(j0, ip);
    float4 p1 = Grad4(j1.x, ip);
    float4 p2 = Grad4(j1.y, ip);
    float4 p3 = Grad4(j1.z, ip);
    float4 p4 = Grad4(j1.w, ip);

    // Normalise gradients
    float4 norm = TaylorInvSqrt(float4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    p4 *= TaylorInvSqrt(dot(p4, p4));

    // Mix contributions from the five corners
    float3 m0 = max(0.6 - float3(dot(x0, x0), dot(x1, x1), dot(x2, x2)), 0.0);
    float2 m1 = max(0.6 - float2(dot(x3, x3), dot(x4, x4)), 0.0);
    m0 = m0 * m0;
    m1 = m1 * m1;
    return 49.0 * (dot(m0 * m0, float3(dot(p0, x0), dot(p1, x1), dot(p2, x2)))
        + dot(m1 * m1, float2(dot(p3, x3), dot(p4, x4))));
}

/**
 * \brief 3D Simplex Noise based Fractal Noise (FBM)
 * \param v Input Coordinate
 * \return [-1.0, 1.0]
 */
float FractalNoise(float2 v)
{
    float value = 0.0;
    float amplitude = 0.5;
    float sum = 0.0;

    [unroll]
    for (int i = 0; i < 4; i++)
    {
        value += amplitude * SimplexNoise(v);
        v *= 2.0;
        amplitude *= 0.5;
        sum += amplitude;
    }
    return value / sum;
}

/**
 * \brief 3D Simplex Noise based Fractal Noise (FBM)
 * \param v Input Coordinate
 * \return [-1.0, 1.0]
 */
float FractalNoise(float3 v)
{
    float value = 0.0;
    float amplitude = 0.5;
    float sum = 0.0;

    [unroll]
    for (int i = 0; i < 4; i++)
    {
        value += amplitude * SimplexNoise(v);
        v *= 2.0;
        amplitude *= 0.5;
        sum += amplitude;
    }
    return value / sum;
}

/**
 * \brief 3D Simplex Noise based Fractal Noise (FBM)
 * \param v Input Coordinate
 * \param harmonics Harmonics
 * \return [-1.0, 1.0]
 */
float FractalNoise(float3 v, int harmonics)
{
    float value = 0.0;
    float amplitude = 0.5;
    float sum = 0.0;

    for (int i = 0; i < harmonics; i++)
    {
        value += amplitude * SimplexNoise(v);
        v *= 2.0;
        amplitude *= 0.5;
        sum += amplitude;
    }
    return value / sum;
}

/**
 * \brief 4D Simplex Noise based Fractal Noise (FBM)
 * \param v Input Coordinate
 * \return [-1.0, 1.0]
 */
float FractalNoise(float4 v, int harmonics)
{
    float value = 0.0;
    float amplitude = 0.5;
    float sum = 0.0;
    
    for (int i = 0; i < harmonics; i++)
    {
        value += amplitude * SimplexNoise(v);
        v *= 2.0;
        amplitude *= 0.5;
        sum += amplitude;
    }
    return value / sum;
}


/**
 * \brief Quaternion multiplication
 */
float4 QMul(float4 q1, float4 q2)
{
    return float4(
        q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
        q1.w * q2.w - dot(q1.xyz, q2.xyz)
    );
}

/**
 * \brief Vector rotation with a quaternion
 */
float3 RotateVector(float3 v, float4 r)
{
    float4 r_c = r * float4(-1, -1, -1, 1);
    return QMul(r, QMul(float4(v, 0), r_c)).xyz;
}

/**
 * \brief Vector rotation with a radian
 */
inline float2 RotateVector(float2 v, float r)
{
    return float2(
        v.x * cos(r) - v.y * sin(r),
        v.x * sin(r) + v.y * cos(r)
    );
}

/**
 * \brief A given angle of rotation about a given axis
 */
float4 RotateAngleAxis(float angle, float3 axis)
{
    float sn = sin(angle * 0.5);
    float cs = cos(angle * 0.5);
    return float4(axis * sn, cs);
}

/**
 * \brief https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
 */
float4 FromToRotation(float3 v1, float3 v2)
{
    float4 q;
    float d = dot(v1, v2);
    if (d < -0.999999)
    {
        float3 right = float3(1, 0, 0);
        float3 up = float3(0, 1, 0);
        float3 tmp = cross(right, v1);
        if (length(tmp) < 0.000001)
        {
            tmp = cross(up, v1);
        }
        tmp = normalize(tmp);
        q = RotateAngleAxis(PI, tmp);
    } else if (d > 0.999999) {
        q = QUATERNION_IDENTITY;
    } else {
        q.xyz = cross(v1, v2);
        q.w = 1 + d;
        q = normalize(q);
    }
    return q;
}

float4 QConj(float4 q)
{
    return float4(-q.x, -q.y, -q.z, q.w);
}

/**
 * \brief https://jp.mathworks.com/help/aeroblks/quaternioninverse.html
 */
float4 QInverse(float4 q)
{
    float4 conj = QConj(q);
    return conj / (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
}

float4 QDiff(float4 q1, float4 q2)
{
    return q2 * QInverse(q1);
}

float4 QLookAt(float3 forward, float3 up)
{
    float3 right = normalize(cross(forward, up));
    up = normalize(cross(forward, right));

    float m00 = right.x;
    float m01 = right.y;
    float m02 = right.z;
    float m10 = up.x;
    float m11 = up.y;
    float m12 = up.z;
    float m20 = forward.x;
    float m21 = forward.y;
    float m22 = forward.z;

    float num8 = (m00 + m11) + m22;
    float4 q = QUATERNION_IDENTITY;
    if (num8 > 0.0)
    {
        float num = sqrt(num8 + 1.0);
        q.w = num * 0.5;
        num = 0.5 / num;
        q.x = (m12 - m21) * num;
        q.y = (m20 - m02) * num;
        q.z = (m01 - m10) * num;
        return q;
    }

    if ((m00 >= m11) && (m00 >= m22))
    {
        float num7 = sqrt(((1.0 + m00) - m11) - m22);
        float num4 = 0.5 / num7;
        q.x = 0.5 * num7;
        q.y = (m01 + m10) * num4;
        q.z = (m02 + m20) * num4;
        q.w = (m12 - m21) * num4;
        return q;
    }

    if (m11 > m22)
    {
        float num6 = sqrt(((1.0 + m11) - m00) - m22);
        float num3 = 0.5 / num6;
        q.x = (m10 + m01) * num3;
        q.y = 0.5 * num6;
        q.z = (m21 + m12) * num3;
        q.w = (m20 - m02) * num3;
        return q;
    }

    float num5 = sqrt(((1.0 + m22) - m00) - m11);
    float num2 = 0.5 / num5;
    q.x = (m20 + m02) * num2;
    q.y = (m21 + m12) * num2;
    q.z = 0.5 * num5;
    q.w = (m01 - m10) * num2;
    return q;
}

float4 QSlerp(in float4 a, in float4 b, float t)
{
    // if either input is zero, return the other.
    if (length(a) == 0.0)
    {
        if (length(b) == 0.0)
        {
            return QUATERNION_IDENTITY;
        }
        return b;
    }
    else if (length(b) == 0.0)
    {
        return a;
    }

    float cosHalfAngle = a.w * b.w + dot(a.xyz, b.xyz);

    if (cosHalfAngle >= 1.0 || cosHalfAngle <= -1.0)
    {
        return a;
    }
    else if (cosHalfAngle < 0.0)
    {
        b.xyz = -b.xyz;
        b.w = -b.w;
        cosHalfAngle = -cosHalfAngle;
    }

    float blendA;
    float blendB;
    if (cosHalfAngle < 0.99)
    {
        // do proper slerp for big angles
        float halfAngle = acos(cosHalfAngle);
        float sinHalfAngle = sin(halfAngle);
        float oneOverSinHalfAngle = 1.0 / sinHalfAngle;
        blendA = sin(halfAngle * (1.0 - t)) * oneOverSinHalfAngle;
        blendB = sin(halfAngle * t) * oneOverSinHalfAngle;
    }
    else
    {
        // do lerp if angle is really small.
        blendA = 1.0 - t;
        blendB = t;
    }

    float4 result = float4(blendA * a.xyz + blendB * b.xyz, blendA * a.w + blendB * b.w);
    if (length(result) > 0.0)
    {
        return normalize(result);
    }
    return QUATERNION_IDENTITY;
}

float4x4 QToMatrix(float4 quat)
{
    float4x4 m = float4x4(float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0));

    float x = quat.x, y = quat.y, z = quat.z, w = quat.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    m[0][0] = 1.0 - (yy + zz);
    m[0][1] = xy - wz;
    m[0][2] = xz + wy;

    m[1][0] = xy + wz;
    m[1][1] = 1.0 - (xx + zz);
    m[1][2] = yz - wx;

    m[2][0] = xz - wy;
    m[2][1] = yz + wx;
    m[2][2] = 1.0 - (xx + yy);

    m[3][3] = 1.0;

    return m;
}

float SdCircle( float2 p, float r )
{
    return length(p) - r;
}

float SdRoundedBox( in float2 p, in float2 b, in float4 r )
{
    r.xy = (p.x>0.0)?r.xy : r.zw;
    r.x  = (p.y>0.0)?r.x  : r.y;
    float2 q = abs(p)-b+r.x;
    return min(max(q.x,q.y),0.0) + length(max(q,0.0)) - r.x;
}

float SdBox( in float2 p, in float2 b )
{
    float2 d = abs(p)-b;
    return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

float SdEquilateralTriangle( in float2 p, in float r )
{
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r/k;
    if( p.x+k*p.y>0.0 ) p = float2(p.x-k*p.y,-k*p.x-p.y)/2.0;
    p.x -= clamp( p.x, -2.0*r, 0.0 );
    return -length(p)*sign(p.y);
}

float SdPentagon( in float2 p, in float r )
{
    const float3 k = float3(0.809016994,0.587785252,0.726542528);
    p.x = abs(p.x);
    p -= 2.0*min(dot(float2(-k.x,k.y),p),0.0)*float2(-k.x,k.y);
    p -= 2.0*min(dot(float2( k.x,k.y),p),0.0)*float2( k.x,k.y);
    p -= float2(clamp(p.x,-r*k.z,r*k.z),r);    
    return length(p)*sign(p.y);
}

float SdHexagon( in float2 p, in float r )
{
    const float3 k = float3(-0.866025404,0.5,0.577350269);
    p = abs(p);
    p -= 2.0*min(dot(k.xy,p),0.0)*k.xy;
    p -= float2(clamp(p.x, -k.z*r, k.z*r), r);
    return length(p)*sign(p.y);
}

float SdOctogon( in float2 p, in float r )
{
    const float3 k = float3(-0.9238795325, 0.3826834323, 0.4142135623 );
    p = abs(p);
    p -= 2.0*min(dot(float2( k.x,k.y),p),0.0)*float2( k.x,k.y);
    p -= 2.0*min(dot(float2(-k.x,k.y),p),0.0)*float2(-k.x,k.y);
    p -= float2(clamp(p.x, -k.z*r, k.z*r), r);
    return length(p)*sign(p.y);
}

float SdCutDisk( in float2 p, in float r, in float h )
{
    float w = sqrt(r*r-h*h); // constant for any given shape
    p.x = abs(p.x);
    float s = max( (h-r)*p.x*p.x+w*w*(h+r-2.0*p.y), h*p.x-w*p.y );
    return (s<0.0) ? length(p)-r :
           (p.x<w) ? h - p.y     :
                     length(p-float2(w,h));
}

float SdMoon(float2 p, float d, float ra, float rb )
{
    p.y = abs(p.y);
    float a = (ra*ra - rb*rb + d*d)/(2.0*d);
    float b = sqrt(max(ra*ra-a*a,0.0));
    if( d*(p.x*b-p.y*a) > d*d*max(b-p.y,0.0) )
        return length(p-float2(a,b));
    return max( (length(p          )-ra),
               -(length(p-float2(d,0))-rb));
}