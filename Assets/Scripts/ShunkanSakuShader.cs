using System;
using UnityEngine;
using UnityEngine.Rendering;

public class ShunkanSakuShader : MonoBehaviour
{
    [SerializeField]
    private Material _shunkanSakuMaterial;

    [SerializeField] private float _pixelPerMeter = 750f;

    private RenderTexture _mainRt, _backBufferRt;
    
    [SerializeField]
    private Material _quadMaterial;
    
    private void Start()
    {
        _quadMaterial = GetComponent<Renderer>().sharedMaterial;
    }

    // Update is called once per frame
    private void Update()
    {
        Vector2Int resolution = new Vector2Int((int)(transform.localScale.x * _pixelPerMeter), (int)(transform.localScale.y * _pixelPerMeter));
        if (_mainRt == null || (_mainRt.width != resolution.x || _mainRt.height != resolution.y))
        {
            _mainRt?.Release();
            _mainRt = new RenderTexture(resolution.x, resolution.y, 0, RenderTextureFormat.ARGBFloat)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0
            };
            
        }
        
        if (_backBufferRt == null || (_backBufferRt.width != resolution.x || _backBufferRt.height != resolution.y))
        {
            _backBufferRt?.Release();
            _backBufferRt = new RenderTexture(resolution.x, resolution.y, 0, RenderTextureFormat.ARGBFloat)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0
            };
        }
        
        _shunkanSakuMaterial.SetVector("_Resolution", new Vector4(resolution.x, resolution.y, 0, 0));
        _shunkanSakuMaterial.SetVector("_MousePos", new Vector4(Input.mousePosition.x / Screen.width, Input.mousePosition.y / Screen.height, 0, Input.GetMouseButton(0) ? 1 : 0));
        _shunkanSakuMaterial.SetFloat("_Aspect", resolution.x / (float)resolution.y);
        _shunkanSakuMaterial.SetTexture("_BackBuffer", _backBufferRt);
        Graphics.Blit(null, _mainRt, _shunkanSakuMaterial, 0);
        Graphics.Blit(_mainRt, _backBufferRt);
        _quadMaterial.SetTexture("_BaseMap", _mainRt);
        
    }

    private void OnDestroy()
    {
        _mainRt?.Release();
        _backBufferRt?.Release();
    }
}