Shader "Unlit/Day01"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
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

            float rBox( in float2 p, in float2 b, in float4 r )
            {
                r.xy = (p.x>0.0)?r.xy : r.zw;
                r.x  = (p.y>0.0)?r.x  : r.y;
                float2 q = abs(p)-b+r.x;
                return min(max(q.x,q.y),0.0) + length(max(q,0.0)) - r.x;
            }

            float Rand(float n){return frac(sin(n) * 43758.5453123);}

            float time;
            float2 resolution;
            float2 mouse;
            float3 spectrum;

            sampler2D texture0;
            sampler2D texture1;
            sampler2D texture2;
            sampler2D texture3;
            sampler2D prevFrame;
            sampler2D prevPass;

            float3 key(float2 uv, float2 id, float t)
            {
                uv.x = uv.x + uv.x * (uv.y+0.1) * 0.1-0.1;
                t = t % 2 -1;
                float lt = saturate(pow(1 - t * t, 200));
                float2 moved = uv - float2(0.0, - lt * 0.1);
                float3 col = rBox(uv - float2(0.0, 0.0), float2(0.5, 0.4), 0.1) < 0. ? float4(0.5,0.5,0.5,1) : 0.03;
                col = rBox(moved - float2(0.0, 0.3), float2(0.505, 0.4), 0.1) < 0. ? float4(0.8,0.8,0.8,1) : col;
                col -= (-0.35 < uv.x && uv.x < -0.1 && 0.4 < uv.y+ lt * 0.1 && uv.y+ lt * 0.1 < 0.6) ? step(Rand(floor(moved.x*20 + id.x*198.23)+floor(moved.y*15 + id.y * 2)), 0.5) : 0;
                col *= col.r > 0.4 ? float3(Rand(id.x*10) *0.8, Rand(id.y) * 0.7, Rand(id.x*20+floor(time*2))+0.1)+0.5 : 1;
                return col;
            }

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            float4 frag (v2f input) : SV_Target
            {
                float2 uv0 = input.uv * 2.0 - 1.0;
                uv0.x *= 10.0 / 6.0;
                float2 uv = uv0;
                float time = _Time.y;

                float3 col;
                uv.y += 1.1;
                uv.x += time * 0.25;
                float2 id2 = floor(uv * 2);
                float2 id = floor(uv * 2 - float2((id2.y) % 2.0 < 1.0 ? 5.5 : 0.0, 0.0));
                uv = frac(uv * 2 - float2(id.y % 2.0 < 1.0 ? 0.5 : 0.0, 0.0));

                col = key(uv * 1.2 - 0.5, id, time + Rand(id.x) *10 + Rand(id.y * 10) * 3);

                return float4(col, 1.0);
            }
            ENDHLSL
        }
    }
}
