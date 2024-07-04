Shader "Unlit/Day02"
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

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            float Torii(float2 uv)
            {
                float res = 0;
                float2 toriiUv = float2(abs(uv.x), uv.y);
                res = SdBox(toriiUv * float2(1 - toriiUv.y, 1 - toriiUv.x * 0.3), float2(0.6, 0.07)) < 0.0 ? float3(0.0, 0.0, 0.0) : 1;
                res = SdBox(toriiUv + float2(0, 0.3), float2(0.5, 0.04)) < 0.0 ? 0.0 : res;
                res = SdBox(toriiUv + float2(0, 0.2), float2(0.06, 0.13)) < 0.0 ? 0.0 : res;
                float2 rotatedUv = RotateVector(toriiUv, -0.04);
                res = SdBox(rotatedUv + float2(-0.3, 0.5), float2(0.05, 0.5)) < 0.0 ? 0.0 : res;

                return res;
            }

            float sin01(float x)
            {
                return sin(x) * 0.5 + 0.5;
            }

            float2 clampm1p1(float2 x)
            {
                return saturate(x * 0.5 + 0.5) * 2.0 - 1.0;
            }
            
            float clampm1p1(float x)
            {
                return saturate(x * 0.5 + 0.5) * 2.0 - 1.0;
            }

            float4 frag (v2f input) : SV_Target
            {
                float2 uv0 = input.uv * 2.0 - 1.0;
                uv0.x *= 10.0 / 6.0;
                float2 uv = uv0;

                // mirror
                float mirrorThreshold = -0.2 + SimplexNoise(float2(uv0.x * 4, _Time.y)) * 0.01;
                if (uv.y < mirrorThreshold)
                {
                    uv.y = -0.35-uv.y;
                    uv += FractalNoise(float3(uv * 4, _Time.y)) * 0.015;
                }
                
                float3 col = saturate((uv.y + 0.4) * float3(0.01, 0, 0.05));

                // clouds
                float2 cloudUv = float2(uv.x, -uv.y) * float2(0.8, 1) + float2(0.0, 0.9);
                float2 cloudPolar = float2(atan2(cloudUv.x, cloudUv.y) / 6.2831, length(cloudUv) * 2.0);
                col += saturate(FractalNoise(float3(cloudPolar * float2(6, 2) + float2(_Time.y * 0.02, 0.0), _Time.x * 0.1)) - 0.2) > 0.3 ? float3(0.02, 0.05, 0.2) : 0.0;

                // moon
                float2 moonUv = uv + float2(0.0, -0.8);
                float3 moonCol = float3(0.0, 0.1, 0.6);
                float moonSd = SdCircle(moonUv, 0.5);
                col += 1 / moonSd * (moonCol + float3(0.02, 0.1, 0.05)) * exp(-moonSd * 20) * 0.85 * (SimplexNoise(float2(_Time.y * 0.2, moonUv.x)) * 0.5 + 0.7);
                col = moonSd < 0.05 ? moonCol * (0.2 + step(saturate(FractalNoise(float3(moonUv * 5, 0.0))), 0.1)) : moonSd < 0.1 ? moonCol : col;

                // torii
                int i;
                float2 toriiUv = uv * 1.2 - float2(0.0, 0.2);
                for (i = 6; i > 0; i--)
                {
                    col = Torii(clampm1p1(toriiUv * pow(i, 1.3)) - float2(0.0, 0.5 - 0.05 * i)) <= 0.0 ? float3(0.0, 0.0, 0.01 * (i-1)) : col;
                }

                // mirror
                if (uv0.y < mirrorThreshold)
                {
                    col *= 0.25;
                }

                // vinnette
                col *= 1.2 - exp(length(uv0) * 0.5) * 0.5;
                col *= 4;

                // noise
                col *= saturate(Pcg2d01((uint2)((uv0 + 21 ) * 1000)).x + 0.5);

                return float4(col, 1.0);
            }
            ENDHLSL
        }
    }
}
