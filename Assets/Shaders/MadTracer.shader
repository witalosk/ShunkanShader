Shader "Unlit/MadTracer"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags
        {
            "RenderType"="Opaque"
        }
        LOD 100

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"
            #include "Common.hlsl"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            struct Surface
            {
                float distace;
                float roughness; // 0.0 - 1.0: roughness of the surface, > 1.0: emission 
                float color;
                bool transparent;
            };

            Surface GetSurface(float distance, float roughness, float color, bool transparent)
            {
                Surface s;
                s.distace = distance;
                s.roughness = roughness;
                s.color = color;
                s.transparent = transparent;
                return s;
            }

            void GetCloserSurface(inout Surface d, float distance, float roughness, float color, bool transparent)
            {
                if (distance < d.distace)
                {
                    d = GetSurface(distance, roughness, color, transparent);
                }
            }

            float3 GetColor(Surface s)
            {
                return step(1., s.roughness) * (
                    s.color < 1.0 ? float3(1, 1, 1) :
                    s.color < 2.0 ? float3(0, 1, 0) :
                    s.color < 3.0 ? float3(1, 0, 1) :
                    s.color < 4.0 ? float3(1, 1, 0) :
                    s.color < 5.0 ? float3(0, 1, 1) :
                    s.color < 6.0 ? float3(1, 0, 0) :
                    float3(0, 0, 1)
                );
            }

            // 3D noise function (IQ, Shane)
            float Noise(float3 p)
            {
                float3 ip = floor(p);
                p -= ip;
                float3 s = float3(7, 157, 113);
                float4 h = float4(0., s.yz, s.y + s.z) + dot(ip, s);
                p = p * p * (3. - 2. * p);
                h = lerp(frac(sin(h) * 43758.5), frac(sin(h + s.x) * 43758.5), p.x);
                h.xy = lerp(h.xz, h.yw, p.y);
                return lerp(h.x, h.y, p.z);
            }

            // hemisphere hash function based on a hash by Slerpy
            float3 HashHs(float3 n, float seed)
            {
                float a = frac(sin(seed) * 43758.5) * 2. - 1.;
                float b = 6.283 * frac(sin(seed) * 41758.5) * 2. - 1.;
                float c = sqrt(1. - a * a);
                float3 r = float3(c * cos(b), a, c * sin(b));
                return r;
            }

            float SdBox(float2 p)
            {
                p = abs(p);
                return max(p.x, p.y);
            }

            void pR(inout float2 p, float a)
            {
                p = cos(a) * p + sin(a) * float2(p.y, -p.x);
            }


            Surface Scene(float3 p)
            {
                float3 q = p;
                Surface d = GetSurface(1, 0., 0., true);
                // float floornoise = .8 * Noise(3. * p + 2.3 * _Time.y) + 0.1 * Noise(20. * p + 2.2 * _Time.y);
                GetCloserSurface(d, min(5. - p.z, 1.5 + p.y), 0.1 + 0.3 * step(fmod(4. * p.z, 1.), .5), .0, false);
                GetCloserSurface(d, length(p + float3(0., 0., 1.9 + sin(_Time.y))) - .500, .99, 1., true);
                pR(q.xy, 0.6 * _Time.y);

                GetCloserSurface(d, length(q + float3(0, 0., 1.9 + sin(_Time.y))) - .445 - 0.09 * sin(43. * q.x - q.y + 10. * _Time.y), 1., 0.1, true);
                if (_Time.y > 24.)p.y -= 0.1 * _Time.y - 2.4;
                q = abs(p - round(p - .5) - .5);
                if (_Time.y > 24.)p.y += 0.1 * _Time.y - 2.4;

                // Lattice (by Slerpy)
                float g = min(min(SdBox(q.xy), SdBox(q.xz)), SdBox(q.yz)) - .05;
                float c = min(.6 - abs(p.x + p.z), .45 - abs(p.y));
                if (_Time.y > 12.) GetCloserSurface(d, max(g, c), 0.1, -0.9, true);

                // Back Boxes
                if (_Time.y > 18.) GetCloserSurface(d, SdBox(p.zx + float2(2, 2)) - .5, 1., 6.5, true);
                if (_Time.y > 17.3) GetCloserSurface(d, SdBox(p.zx + float2(2, -2)) - .5, 1., 2.5, true);
                
                return d;
            }


            float3 SceneNormal(float3 p)
            {
                float m = Scene(p).distace;
                float2 e = float2(0, 0.05);
                return normalize(m - float3(Scene(p - e.yxx).distace, Scene(p - e.xyx).distace, Scene(p - e.xxy).distace));
            }


            float3 MadTracer(in float3 camPos, in float3 dir, in float seed)
            {
                float3 result = 0.0;
                float rayCorr = 0.0, reflectedCorr = 0.0;
                Surface s;
                float3 camPos0 = camPos;
                float3 dir0 = dir;
                s.distace = 0.0;
                for (int i = 0; i < 140; i++)
                {
                    seed = frac(seed + _Time.y * float(i + 1) + .1);
                    
                    camPos = lerp(camPos0, HashHs(camPos, seed), 0.002); // antialiasing
                    dir = lerp(dir0, HashHs(dir0, seed), 0.06 * s.distace); // antialiasing
                    
                    float3 currentPos = camPos + dir * rayCorr;
                    float3 normal = SceneNormal(currentPos); // normal of new origin
                    s = Scene(currentPos);
                    rayCorr += s.transparent ? 0.25 * abs(s.distace) + 0.0008 : 0.25 * s.distace; // カラーが0の時は不透明な物体として扱う
                    result += 0.007 * GetColor(s);

                    // reflection
                    seed = frac(seed + _Time.y * float(i + 2) + 0.1);
                    float3 reflectedDir = lerp(reflect(dir, normal), HashHs(normal, seed), s.roughness); // reflect depending on roughness
                    Surface reflected = Scene(currentPos + reflectedDir * reflectedCorr);
                    // Surface reflected = Scene(currentPos + reflectedDir * (Noise(currentPos + seed) + 1));
                    reflectedCorr += reflected.transparent ? 0.25 * abs(reflected.distace) : 0.25 * reflected.distace;
                    result += 0.007 * GetColor(reflected);
                }

                return result;
            }

            float4 frag(v2f input) : SV_Target
            {
                float2 uv = input.uv;
                float2 fragCoord = uv * _ScreenParams.xy;
                float4 col = 0.0;

                float seed = sin(fragCoord.x + fragCoord.y) * sin(fragCoord.x - fragCoord.y);
                // float3 bufa= texture(iChannel0, uv).xyz;

                // camera
                float3 ro, rd;
                float2 uv2 = (2. * fragCoord.xy - _ScreenParams.xy) / _ScreenParams.xy.x;
                ro = float3(0, 0, -5);
                rd = normalize(float3(uv2, 1));
                // // rotate scene
                // if (_Time.y > 12.)
                // {
                //     pR(rd.xz, .5 * -sin(.17 * _Time.y));
                //     pR(rd.yz, .5 * sin(.19 * _Time.y));
                //     pR(rd.xy, .4 * -cos(.15 * _Time.y));
                // }
                // render    
                col.xyz = MadTracer(ro, rd, seed);

                // float fade = min(3. * abs(sin((3.1415 * (_Time.y - 12.) / 24.))), 1.);
                // fragColor =clamp(float4(0.7*scol+0.7*bufa, 0.)*fade, 0., 1.); // with blur

                return saturate(col);
            }
            ENDHLSL
        }
    }
}