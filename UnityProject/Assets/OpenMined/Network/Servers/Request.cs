using UnityEngine;
using System.Collections;
using UnityEngine.Networking;
using System.Collections.Generic;
using System;
using OpenMined.Network.Utils;
using OpenMined.Hex.HexConvertors.Extensions;

namespace OpenMined.Network.Servers
{
    public class Request
    {
        public class EthResponse
        {
            public string jsonrpc;
            public int id;
        }

        public class BlockNumber : EthResponse
        {
            public string result;
        }

        public class Call : EthResponse
        {
            public string result;
        }
        
        public class GetModelResponse
        {
            public String address = "";
            public Int32 bounty;
            public Int32 initialError;
            public Int32 targetError;
            public List<String> hexAddress;
            
            public String configAddress = "";

            int numParameters = 5;
            readonly List<Type> types;

            public GetModelResponse(string hexString)
            {
                hexAddress = new List<String>();
                types = new List<Type>
                {
                    address.GetType(),
                    bounty.GetType(),
                    initialError.GetType(),
                    targetError.GetType(),
                    hexAddress.GetType()
                };

                var objects = EthereumAbiUtil.GetParametersHex(hexString, numParameters, types);

                address = (String)objects[0];
                bounty = (Int32)objects[1];
                initialError = (Int32)objects[2];
                targetError = (Int32)objects[3];
                hexAddress = (List<String>)objects[4];
                MakeIPFSHash();
            }

            private void MakeIPFSHash()
            {
                var firstHalf = hexAddress[0].HexToUTF8String();
                var secondHalf = hexAddress[1].HexToUTF8String();

                configAddress = firstHalf + secondHalf;
                configAddress = configAddress.Substring(0, 46);
            }
        }

        public static string identityURL = "http://localhost:3000/";

        public static string contractAddress = "0xd60e1a150b59a89a8e6e6ff2c03ffb6cb4096205";
        public static string infuraURL = "https://api.infura.io/v1/jsonrpc/";
        public static string infuraNetwork = "rinkeby/";

        public int numModels;
        public GetModelResponse modelResponse;

        public Coroutine Coroutine { get; private set; }
        public object result;
        private IEnumerator target;

        public Request() { }

        public Request(MonoBehaviour owner, IEnumerator target)
        {
            this.target = target;
            this.Coroutine = owner.StartCoroutine(Run());
        }

        private IEnumerator Run()
        {
            while (target.MoveNext())
            {
                result = target.Current;
                yield return result;
            }
        }

        public IEnumerator GetIdentity(string method,
                                              string modelAddress = "")
        {
            string URL = identityURL;
            
            if(method.Length > 0)
            {
                var model = WWW.EscapeURL(modelAddress);
                URL += "/" + method + "?model=" + model;
            }

            Debug.LogFormat("Request.GetIdentity {0}", URL);
            UnityWebRequest www = UnityWebRequest.Get(URL);
            www.SetRequestHeader("accept", "text/plain");

            yield return www.SendWebRequest();

            if (www.isNetworkError || www.isHttpError)
            {
                Debug.Log(www.error);
                yield return null;
            }
            else
            {
                yield return www.downloadHandler.text;
            }
        }

        public IEnumerator Get<T>(string method, string data = "") where T : EthResponse
        {
            string URL = infuraURL + infuraNetwork + method;

            if (data != "") {
                URL += "?params=" + data;
            }

            Debug.LogFormat("Request.Get {0}", URL);
            UnityWebRequest www = UnityWebRequest.Get(URL);
            www.SetRequestHeader("accept", "application/json");

            yield return www.SendWebRequest();

            if (www.isNetworkError || www.isHttpError)
            {
                Debug.Log(www.error);
                yield return null;
            }
            else
            {
                string json = www.downloadHandler.text;

                T response = JsonUtility.FromJson<T>(json);
                yield return response;
            }
        }

        public IEnumerator GetBlockNumber(MonoBehaviour owner)
        {
            Request req = new Request(owner, Get<BlockNumber>("eth_blockNumber"));
            yield return req.Coroutine;

            Request.BlockNumber response = req.result as BlockNumber;
            int res = (int) new System.ComponentModel.Int32Converter().ConvertFromString(response.result);
            Debug.LogFormat("\nCurrent Rinkeby Block Number: {0}", res.ToString("N"));
        }

        private string EncodeData(string data)
        {
            string encodedData = WWW.EscapeURL("[{\"to\":\"" + contractAddress + "\",\"data\":\"" + data + "\"},\"latest\"]");
            return encodedData;
        }

        public IEnumerator GetNumModels(MonoBehaviour owner)
        {
            // TODO: convert "getNumModels" to hex.
            string data = EncodeData("0x3c320cc2");
            Request req = new Request(owner, Get<Call>("eth_call", data));
            yield return req.Coroutine;

            Call response = req.result as Call;
            numModels = (int)new System.ComponentModel.Int32Converter().ConvertFromString(response.result);
            Debug.LogFormat("\nNum Models: {0}", numModels.ToString("N"));
        }

        public IEnumerator GetModel(MonoBehaviour owner)
        {
            yield return GetNumModels(owner);
       
            var keccak = new Sha3Keccak();
            var d = keccak.CalculateHash("getModel(uint256)");
            d = d.Substring(0, 8);

            // get latest model
            var value = EthereumAbiUtil.EncodeInt32(numModels - 1);

            string data = EncodeData("0x" + d + value);

            Request req = new Request(owner, Get<Call>("eth_call", data));
            yield return req.Coroutine;

            Call response = req.result as Call;

            modelResponse = new GetModelResponse(response.result);

            Debug.LogFormat("Model {0}, {1}. {2}, {3}, {4}", modelResponse.address, 
                            modelResponse.bounty, modelResponse.initialError, modelResponse.targetError, 
                            modelResponse.configAddress);
        }

        public IEnumerator AddModel(MonoBehaviour owner, string ipfsHash)
        {
            Request req = new Request(owner, GetIdentity("addModel", ipfsHash));
            yield return req.Coroutine;

            Debug.LogFormat("response {0}", req.result);
        }
    }
}
