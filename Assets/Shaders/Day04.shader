Shader "Day04"
{
    CGINCLUDE

    #include "Common.hlsl"
    #include "UnityCG.cginc"

    #define time _Time.y

    sampler2D _MainTex;
    float4 _MainTex_ST;
    float4 _MainTex_TexelSize;

    sampler2D _BackBuffer;
    float _Aspect;
    float4 _MousePos;
    float2 _Resolution;

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

    static float r3 = 1.73205080756;    // sqrt(3)の値
    static float r3i = 0.57735026919;   // 1/sqrt(3)の値

    // 直交<->斜交の座標変換行列
    static float2x2 tri2cart = float2x2(1., .5, 0., r3 * .5);
    static float2x2 cart2tri = float2x2(1., -r3i, 0., r3i * 2.);

    float4 TriCoordinate(float2 uv) {
       uv = mul(cart2tri, uv);
       float2 index = floor(uv);    //整数部分
       float2 pos = frac(uv);       //小数部分
       
       // 三角形の重心座標を計算
       index += (2. - step(pos.x + pos.y, 1.)) / 3.;
       
       return float4(mul(tri2cart, index), pos);
    }

    float2 InvTriCoordinate(float4 coord) {
        float2 index = coord.xy;  // 三角形の座標部分
        float2 pos = coord.zw;    // 小数部分
        
        // 三角形の重心座標から整数部分を計算
        index -= (2. - step(pos.x + pos.y, 1.)) / 3.;
        
        // カート座標に変換
        float2 uv = index + pos;
        uv = mul(tri2cart, uv);  // 三角形からカート座標系に変換
        
        return uv;
    }

    inline float4 GetBackBufferByUv(float2 uv)
    {
        return tex2D(_BackBuffer, uv);
    }

    v2f Vert (appdata v)
    {
        v2f o;
        o.vertex = UnityObjectToClipPos(v.vertex);
        o.uv = TRANSFORM_TEX(v.uv, _MainTex);
        return o;
    }
    // #define GridSize (0.005 * (1 + (int)(_Time.y * 4) % 6))
    #define GridSize 0.01

    float2 GetGridCoord(float2 uv)
    {
        return floor(uv / GridSize);
    }

    float2 GetInGridCoord(float2 uv)
    {
        return frac(uv / GridSize);
    }

    float2 GetUv(float2 gridCoord, float2 inGridCoord)
    {
        return (gridCoord + inGridCoord) * GridSize;
    }
    

    float4 GetValue(float2 gridCoord, float2 inGridCoord)
    {
        return GetBackBufferByUv(GetUv(gridCoord, inGridCoord) / float2(_Aspect, 1));
    }

    float Rand(float2 co){
        return frac(sin(dot(co.xy ,float2(12.9898,78.233))) * 43758.5453);
    }

    float4 FragMain(v2f i) : SV_Target
    {
        float2 uv0 = i.uv;
        float2 uv = uv0 * float2(_Aspect, 1);

        float2 gridCoord = GetGridCoord(uv);
        float2 inGridCoord = GetInGridCoord(uv);

        if (distance(_MousePos, uv) < 0.03 && _MousePos.w > 0.5)
        {
            return float4(ConvertHsvToRgb(float3((_Time.y) % 1.0, 1, 1)), 1);
        }

        float2 idx = floor(uv);
        float2 insideUv = uv % 1.0;
        for (float l = 0.0; l < 6.0; l += 1.0)
        {
            if (Pcg2d01(idx + float2(110.0 + _Time.y, 1024.0)).x < 0.4) break;
            
            idx = idx * 2.0 + floor(insideUv / 0.5);
            insideUv = fmod(insideUv, 0.5) * 2.0;
        }
        // float4 col = float4(ConvertHsvToRgb(float3(Rand(idx) + insideUv.y * 0.15, 0.5 + 0.5 * insideUv.x, 0.7)), 1) * 0.01;
        float4 col = 0;
        if (length(Rand(idx)) > 0.75)
        {
            // gridCoord += Pcg2d01(idx);
            // col += GetBackBufferByUv(uv0 + (2 * float2(Rand(idx), Rand(idx + 100)) - 1) * 0.001);
        }
        
        
        float3 c = GetValue(gridCoord, 0.5).rgb;
        float3 n = GetValue(gridCoord + float2(0, Rand(idx) * 2), 0.5).rgb;
        float3 s = GetValue(gridCoord + float2(0, -1), 0.5).rgb;
        float3 e = GetValue(gridCoord + float2(1, 0), 0.5).rgb;
        float3 w = GetValue(gridCoord + float2(-1, 0), 0.5).rgb;
        float3 ne = GetValue(gridCoord + float2(1, 1), 0.5).rgb;
        float3 nw = GetValue(gridCoord + float2(-1, 1), 0.5).rgb;
        float sum = length(c) + length(n) + length(s) + length(e) + length(w) + length(ne) + length(nw);
        if (sum > 7) return col * 0.5;

        col.rgb += c * 0.98;

        if (length(n) > 0.3 && Pcg3d01(float3(_Time.x, gridCoord)).x > 0.6 * (0.5 + Rand(idx + _Time*20)))
        {
            col.rgb = n;
        }
        
        if (ne.g > 0.5 && Pcg3d01(float3(_Time.x, gridCoord)).y > 0.5)
        {
            col.rgb = ne;
        }
        
        if (nw.b > 0.5 && Pcg3d01(float3(_Time.x, gridCoord)).z > 0.5)
        {
            col.rgb = nw;
        }

        if (e.g > 0.5 && Pcg3d01(float3(_Time.x*2, gridCoord)).y > 0.9)
        {
            col.rgb = ne;
        }
        
        if (w.b > 0.5 && Pcg3d01(float3(_Time.x*3, gridCoord)).z > 0.9)
        {
            col.rgb = nw;
        }

        if (length(col.rgb) > 0.5 && Pcg3d01(float3(_Time.y * 100, gridCoord)).z > 0.9)
        {
            col.rgb = HsvShift(col.rgb, float3(0.1, 1, 1));
        }

        // col *= float4(ConvertHsvToRgb(float3(Rand(idx) + insideUv.y * 0.15, 0.5 + 0.5 * insideUv.x, 0.7)), 1) * 1.5;
        
        // return saturate(col);
        return saturate(col) * (SdEquilateralTriangle(RotateVector(inGridCoord - 0.5, Pcg01((uint)gridCoord.x + 1000 * gridCoord.y) + _Time.y), 0.3 + Pcg01((uint)gridCoord.x + 1000 * gridCoord.y + _Time.y*100)) < 0.0 ? 1 : 0);
    }

    ENDCG

    Properties
    {
        _MainTex("Texture", 2D) = "white" {}
    }

    SubShader
    {
        Tags{ "RenderType" = "Opaque" }

        ZTest Always
        Cull Off
        ZWrite Off
        Blend Off

        // 0
        Pass
        {
            CGPROGRAM
            #pragma target   5.0
            #pragma vertex   Vert
            #pragma fragment FragMain
            ENDCG
        }
    }
}