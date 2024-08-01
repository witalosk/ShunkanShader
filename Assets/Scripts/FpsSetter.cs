using UnityEngine;

public class FpsSetter : MonoBehaviour
{
    [SerializeField] private int _fps = 60;

    private void Awake()
    {
        Application.targetFrameRate = _fps;
        QualitySettings.vSyncCount = 0;
    }
}