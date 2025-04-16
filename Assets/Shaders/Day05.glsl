#version 410 core


#define FLOAT_MAX float(uint(0xffffffffu)) // GLSLでは直接16進数でのfloat初期化は標準ではないためuint経由
#define FLOAT_MAX_INVERT (1.0 / FLOAT_MAX)
#define PI 3.1415926535897932384626433832795
#define QUATERNION_IDENTITY vec4(0.0, 0.0, 0.0, 1.0)



vec3 ConvertRgbToHsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-5; // Epsilon
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 ConvertHsvToRgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}



/**
 * \brief HSVをずらす
 */
vec3 HsvShift(vec3 color, vec3 hsvShift)
{
    vec3 hsv = ConvertRgbToHsv(color);
    hsv.x += hsvShift.x;
    hsv.x = mod(hsv.x, 1.0); // Use mod for float remainder
    hsv.y = hsv.y * hsvShift.y;
    hsv.z = hsv.z * hsvShift.z;
    return ConvertHsvToRgb(hsv);
}


vec4 Mod289(vec4 x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 Mod289(vec3 x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 Mod289(vec2 x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

float Mod289(float x)
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 Permute(vec4 x)
{
    return Mod289(((x * 34.0) + 1.0) * x);
}

vec3 Permute(vec3 x)
{
    return Mod289(((x * 34.0) + 1.0) * x);
}

float Permute(float x)
{
    return Mod289(((x * 34.0) + 1.0) * x);
}

vec4 TaylorInvSqrt(vec4 r)
{
    return inversesqrt(r); // rsqrt -> inversesqrt
}

float TaylorInvSqrt(float r)
{
    return inversesqrt(r); // rsqrt -> inversesqrt
}

vec4 Grad4(float j, vec4 ip)
{
    const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
    vec4 p;

    // Original HLSL uses implicit j.xxx swizzle
    p.xyz = floor(fract(vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
    p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
    vec4 s = vec4(1.0 - step(0.0, p)); // step(0.0, p) is equivalent to (p >= 0.0 ? 1.0 : 0.0)
    p.xyz = p.xyz + (s.xyz * 2.0 - 1.0) * s.www; // Use .www swizzle

    return p;
}

/**
 * \brief 2D Simplex Noise [-1.0, 1.0]
 */
float SimplexNoise(vec2 v)
{
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                       -0.577350269189626,  // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
    // First corner
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);

    // Other corners
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;

    // Permutations
    i = Mod289(i); // Avoid truncation effects in permutation
    vec3 p = Permute(Permute(i.y + vec3(0.0, i1.y, 1.0))
          + i.x + vec3(0.0, i1.x, 1.0));

    vec3 m = max(vec3(0.5) - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3(0.0));
    m = m * m;
    m = m * m;

    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;

    // Normalise gradients implicitly by scaling m
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

    // Compute final noise value at P
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}



/**
 * \brief Vector rotation with a radian (2D)
 */
vec2 RotateVector(vec2 v, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    // mat2 rotationMatrix = mat2(c, -s, s, c); // This is column major: (c, s), (-s, c)
    // return rotationMatrix * v;
    // Direct calculation:
    return vec2(
        v.x * c - v.y * s,
        v.x * s + v.y * c
    );
}




uniform float fGlobalTime; // in seconds
uniform vec2 v2Resolution; // viewport resolution (in pixels)
uniform float fFrameTime; // duration of the last frame, in seconds

uniform sampler1D texFFT; // towards 0.0 is bass / lower freq, towards 1.0 is higher / treble freq
uniform sampler1D texFFTSmoothed; // this one has longer falloff and less harsh transients
uniform sampler1D texFFTIntegrated; // this is continually increasing
uniform sampler2D texPreviousFrame; // screenshot of the previous frame
uniform sampler2D texChecker;
uniform sampler2D texNoise;
uniform sampler2D texTex1;
uniform sampler2D texTex2;
uniform sampler2D texTex3;
uniform sampler2D texTex4;

layout(location = 0) out vec4 out_color; // out_color must be written in order to see anything

vec4 plas( vec2 v, float time )
{
	float c = 0.5 + sin( v.x * 10.0 ) + cos( sin( time + v.y ) * 20.0 );
	return vec4( sin(c * 0.2 + cos(time)), c * 0.15, cos( c * 0.1 + time / .4 ) * .25, 1.0 );
}

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float Rose(float r, float t, float a, float b)
{
  return abs((a * sin(b * t) + (1.0 - a)) - r);
}

float ns(float t)
{
  return sin(t) * 0.5 + 0.5;
}

void main(void)
{
  vec3 col = vec3(1.0);
	vec2 uv = vec2(gl_FragCoord.x / v2Resolution.x, gl_FragCoord.y / v2Resolution.y);
	uv -= 0.5;
	uv /= vec2(v2Resolution.y / v2Resolution.x, 1);
  
  uv = RotateVector(uv, fGlobalTime*0.);

  
  vec2 p = uv; 
  float time = fGlobalTime;
  
  p += SimplexNoise( vec2(time*0.1, time*0.05  + 500.0)) * 0.3;
  
  //p.x += 100.0;
  
  float y = p.y;
  float x = p.x;

  p.x = sqrt(x*x+y*y) * 3.0;
  p.y = atan(y, x);// + sin(time) * 0.3;
  

  
  float wholeR = p.x;
  float wholeT = p.y;
  
  //out_color.xy = mod(p,0.1)*10;
  //return;
  
  p += vec2(ns(time*0.5) * 2);
  
  
  vec2 idx = floor(p);
  vec2 insideUv = mod(p, 1.0);
  
  for(float i = 0.0; i < 6.0; i += 1.0)
  {
    if (rand(idx + vec2(110.0, rand(vec2(10, floor(time*2))))) < 0.5) break;
    
    idx = idx * 2.0 + floor(insideUv / 0.5);
    insideUv = mod(insideUv, 0.5) * 2.0;
  }
  
  vec2 insidePolar;
  x = insideUv.x;
  y = insideUv.y;
  insidePolar.x = sqrt(x*x+y*y);
  insidePolar.y = atan(y,x);
  
  col = vec3(insideUv, rand(idx));
  
 
  float a = idx.x * 0.5 + ns(floor(time));
  float b = idx.y * ns(floor(time * 3)) * 10;
  
  col += Rose(wholeR * 0.3, wholeT, a, b) < 1.5 * (ns(time*0.5) + 0.5) ? 1.0 : 0.0;
  
  col += (insideUv.x - 0.5) < 0.3 ? 1.0 : 0.0;
  
  col = clamp(col, 0, 1);
  
  
  col.gb -= Rose(insidePolar.x, insidePolar.y, idx.x + a, b * ns(time*0.1)*10) < 0.5 ? (insideUv) : vec2(0.0);
  
  
  
	//col.rgb = vec3(1.0) - col;
  
  col *= clamp(pow(1.0 - length(uv), 0.3) * 1.5, 0, 1);
	out_color = vec4(col, 1.0);
}

