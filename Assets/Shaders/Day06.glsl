#version 410 core

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

mat2 rot(float r) {
  return mat2(cos(r), sin(r), -sin(r), cos(r));
}


float sdOctahedron( vec3 p, float s)
{
  p = abs(p);
  return (p.x+p.y+p.z-s)*0.57735027;
}

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float mainDist(vec3 p, float time, float loop)
{
  
  p.xy *= rot(p.z*0.3);
  p.z += time;
  vec3 grid = mod(p, 2.0) - 1.0;
  vec3 id = p - grid;
  
  time += p.z*0.025;
  float t = floor(time) + pow(fract(time*2.0), 0.8);
  float octa = sdOctahedron(grid, sin(time*0.1 + loop*0.5) + 1.5);
  float sp = length(grid) - 0.5 * (1.0 - mod(t, 1.0));
  float octasp = min(octa, sp);
 
  return min(min(min(octasp, sdBox(grid, vec3(0.01,0.01,1.0))), sdBox(grid, vec3(1.0,0.01,0.01))), sdBox(grid, vec3(0.01,1.0,0.01)));
}



void main(void)
{
  float time = fGlobalTime;
  vec3 col = vec3(0.0);
	vec2 uv = vec2(gl_FragCoord.x / v2Resolution.x, gl_FragCoord.y / v2Resolution.y);
	vec2 p = uv * 2.0 - vec2(1.0);
  p.y *= v2Resolution.y / v2Resolution.x;
  
  float radius = 0.5;
  float phi = time * 0.2;
  
  
  vec3 ro = vec3(cos(phi)*radius, 0.0, sin(phi) * radius);
  vec3 target = vec3(0.0);
  vec3 camdir = normalize(target - ro);
  vec3 side = cross(camdir, vec3(0.,1.,0.));
  vec3 up = cross(side, camdir);
  float fov = 0.4;
  
  vec3 rdir = normalize(p.x * side + p.y * up + camdir * fov);
  //rdir = vec3(p, 0.5);
  
  
  
  
  float sum = 0.0;
  vec3 cpos = ro;
  
  
  for (float i = 0.0; i < 60.0; i+=1.0)
  {
    float d = mainDist(cpos, time, i);
    sum += abs(d);
    float stepsize = max(0.04, d) + abs(cpos.y) * 0.01;

    for (float j = 0.0; j < 10.0; j+=1.0)
    {
      float dir = mod(j, 2.0) > 0.5 ? 1.0 : -1.0;
      col += vec3(1.0) * exp(1.0 - sum * 0.2) / stepsize * (cos(vec3(0.2, 1.5, 2.5) + (vec3(time * dir, time * dir, time * 3.0) + cpos) * j) + 0.5);
    }
    col += clamp((vec3(2.0) / abs(d*0.1) * 0.1), vec3(0.0), vec3(150.0)) ;
  
    
    cpos += rdir * stepsize;
    
  }
  
  col = pow(col/4000., vec3(1.8));
  col *= vec3(0.8, 0.9, 1.1);
  
  
  
  
	out_color = vec4(col, 1.0);
}