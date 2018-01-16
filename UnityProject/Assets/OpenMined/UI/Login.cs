using UnityEngine;
using UnityEngine.UI;
using System;
using System.Collections;

using ZXing;
using ZXing.QrCode;

using OpenMined.Network.Servers;

namespace OpenMined.UI
{
    public class Login : MonoBehaviour
    {
        public Button loginButton;

        private static Color32[] Encode(string textForEncoding, int width, int height)
        {
            var writer = new BarcodeWriter
            {
                Format = BarcodeFormat.QR_CODE,
                Options = new QrCodeEncodingOptions
                {
                    Height = height,
                    Width = width
                }
            };
            return writer.Write(textForEncoding);
        }

        public Texture2D GenerateQR(string text)
        {
            var encoded = new Texture2D(256, 256);
            var color32 = Encode(text, encoded.width, encoded.height);
            encoded.SetPixels32(color32);
            encoded.Apply();
            return encoded;
        }

        //private IEnumerator Start()
        //{
        //    Debug.Log("Login OnGUI()");
            
        //    Request r = new Request();

        //    Request req = new Request(this, r.GetIdentity(""));
        //    yield return req.Coroutine;
        //    string URI = req.result as string;
        //    Debug.LogFormat("\nURI: {0}", URI);

        //    Texture2D qrTexture = GenerateQR(URI);
        //    Sprite qrSprite = Sprite.Create(qrTexture, new Rect(0.0f, 0.0f, 256, 256), new Vector2(0.5f, 0.5f), 100.0f);
        //    loginButton.image.sprite = qrSprite;
        //}
    }
}
