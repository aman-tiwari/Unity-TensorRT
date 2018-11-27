using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;

public class UnityTRT : MonoBehaviour
{

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void LoggerDelegate(int severity, string str);

    static void CallBackFunction(int severity, string str) {
        if (severity == 3) Debug.Log("TRT: " + str);
        else if (severity == 2) Debug.LogWarning("TRT: " + str);
        else if (severity == 1) Debug.LogError("TRT: " + str);
    }

    [DllImport("unitytrtcuda")]
    static extern bool loadModel(string path);

    [DllImport("unitytrtcuda")]
    static extern bool dispose();

    [DllImport("unitytrtcuda")]
    static extern void setDebugFunction(System.IntPtr fp);

    [DllImport("unitytrtcuda")]
    static extern bool bindTextures(System.IntPtr inputTexture, System.IntPtr outputTexture);

    [DllImport("unitytrtcuda")]
    static extern bool inferOnTextures();

    // Start is called before the first frame update
    void Start() {
        var inputTexture = new Texture2D(28, 28, TextureFormat.RFloat, false);
        
        var outputTexture = new Texture2D(10, 1, TextureFormat.RFloat, false);
        LoggerDelegate callback_delegate = new LoggerDelegate(CallBackFunction);


        // Convert callback_delegate into a function pointer that can be
        // used in unmanaged code.
        System.IntPtr intptr_delegate =
            Marshal.GetFunctionPointerForDelegate(callback_delegate);

        setDebugFunction(intptr_delegate);

        string path = @"C:\Users\a\Downloads\TensorRT-5.0.4.3.Windows10.x86_64.cuda-9.0.cudnn7.3\TensorRT-5.0.4.3\data\mnist\mnist.onnx";
        Debug.Log(path);
        Debug.Log(loadModel(path));
       // Debug.Log(RegisterInputTexture(inputTexture.GetNativeTexturePtr()));
       // Debug.Log(RegisterOutputTexture(outputTexture.GetNativeTexturePtr()));
        Debug.Log(bindTextures(inputTexture.GetNativeTexturePtr(), outputTexture.GetNativeTexturePtr()));
        Debug.Log(inferOnTextures());
        float t = Time.realtimeSinceStartup;
        for(int i = 0; i < 1000; i++) {
            inferOnTextures();
        }
        float dt = Time.realtimeSinceStartup - t;
        Debug.Log((dt / 1000)*1000);
        Debug.Log(dispose());
    }

    [DllImport("kernel32", SetLastError = true)]
    private static extern bool FreeLibrary(System.IntPtr hModule);

    public static void UnloadImportedDll(string DllPath) {
        foreach (System.Diagnostics.ProcessModule mod in System.Diagnostics.Process.GetCurrentProcess().Modules) {
            if (mod.FileName == DllPath) {
                FreeLibrary(mod.BaseAddress);
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
