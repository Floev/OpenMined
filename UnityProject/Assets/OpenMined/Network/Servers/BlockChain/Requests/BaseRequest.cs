using UnityEngine;
using System.Collections;
using System;
using UnityEngine.Networking;
using Newtonsoft.Json;
using System.IO;
using Newtonsoft.Json.Linq;


namespace OpenMined.Network.Servers.BlockChain.Requests
{
    public abstract class BaseRequest<T>
    {
        public Method method;
        public string path;

        protected string response;
            
        public BaseRequest (Method method, string path)
        {
            this.method = method;
            this.path = path;
        }

        protected UnityWebRequest GetRequest (string subPath = "", WWWForm postBody = null)
        {
            var builder = new UriBuilder();
            builder.Host = GetHost();
            builder.Port = 3000;
            builder.Scheme = "http";
            builder.Path = this.path + subPath;

            switch (this.method)
            {
                case Method.GET:
                    return UnityWebRequest.Get(builder.Uri.AbsoluteUri);
                case Method.POST:
                    var req = UnityWebRequest.Post(builder.Uri.AbsoluteUri, postBody);
                    //req.SetRequestHeader("content-type", "application/json");
                    return req;
            };

            return null;
        }

        public IEnumerator RunRequest ()
        {
            var request = GetWebRequest();
            yield return request.SendWebRequest();
            if (request.isNetworkError || request.isHttpError)
            {
                Debug.Log(request.error);
                yield return null;
            }
            else
            {
                yield return request.downloadHandler.text;
                this.response = request.downloadHandler.text;
            }
        }

        public void RunRequestSync ()
        {
            var request = GetWebRequest();
            request.SendWebRequest();

            while (!request.isDone) {}

            if (request.isNetworkError || request.isHttpError)
            {
                Debug.Log(request.error);
            }
            else
            {
                this.response = request.downloadHandler.text;
            }
        }

        public T GetResponse ()
        {
            return JsonConvert.DeserializeObject<T>(this.response);
        }

        private string GetHost ()
        {
            return Config.Config.bygoneServer;
        }

        abstract public UnityWebRequest GetWebRequest();
    }

    public enum Method
    {
        GET,
        POST
    }
}
