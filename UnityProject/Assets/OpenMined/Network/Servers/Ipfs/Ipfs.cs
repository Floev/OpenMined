using UnityEngine;
using System.Collections;
using UnityEngine.Networking;
using System;
using OpenMined.Syft.Tensor;
using System.Threading.Tasks;

using OpenMined.Network.Controllers;

namespace OpenMined.Network.Servers
{
    public class Ipfs
    {

        public static string POST_URL = "https://ipfs.infura.io:5001/api/v0/add?stream-channels=true";
        public static string GET_URL = "https://ipfs.infura.io/ipfs";

        public IpfsResponse Write<T>(T data)
        {
            var serializedData = JsonUtility.ToJson(data);
            Debug.Log(serializedData);

            /**
             * The blob that has to go over the line.
             * 
             * Basically this is what happens in HTTP when you PUT a file with
             * content-type multipart/form-data
             */
            var stringData = "--------------------------30a67cb5e62650e3\r\nContent-Disposition: form-data; name=\"file\"; filename=\"model\";\r\n";
            stringData    += "Content-Type: application/octet-stream\r\n\r\n";
            stringData    += serializedData + "\r\n";
            stringData    += "--------------------------30a67cb5e62650e3--\r\n";

            var bytes = System.Text.Encoding.UTF8.GetBytes(stringData);
            UnityWebRequest www = UnityWebRequest.Put(Ipfs.POST_URL, bytes);
            www.SetRequestHeader("Content-Type", "multipart/form-data; boundary=------------------------30a67cb5e62650e3");
            var op = www.SendWebRequest();

            while (!op.isDone)
            {
                // wait for operation to finish
            }

            if (www.isHttpError || www.isNetworkError)
            {
                Debug.Log("Error making IPFS request: " + www.error);
                return null;
            }
            else
            {
                string json = www.downloadHandler.text;
                var response = JsonUtility.FromJson<IpfsResponse>(json);
                Debug.Log("Got Ipfs response: " + response);
                return response;
            }    
        }

        public static FloatTensor Get (string path)
        {
            var www = UnityWebRequest.Get(GET_URL + "/" + path);
            var op = www.SendWebRequest();
            while (!op.isDone)
            {
                // wait for operation to finish
            }

            if (www.isHttpError || www.isNetworkError)
            {
                Debug.Log("Error getting IPFS data: " + www.error);
                return null;
            }
            else
            {
                var json = www.downloadHandler.text;
                var tensor = JsonUtility.FromJson<FloatTensor>(json);

                return tensor;
            }
        }

        public static IpfsModel GetModel(string path)
        {
            var www = UnityWebRequest.Get(GET_URL + "/" + path);
            var op = www.SendWebRequest();
            while (!op.isDone)
            {
                // wait for operation to finish
            }

            if (www.isHttpError || www.isNetworkError)
            {
                Debug.Log("Error getting IPFS data: " + www.error);
                return null;
            }
            else
            {
                var json = www.downloadHandler.text;
                return JsonUtility.FromJson<IpfsModel>(json);
            }
        }

    }
}
