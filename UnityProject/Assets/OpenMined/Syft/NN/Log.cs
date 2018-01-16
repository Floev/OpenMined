using System;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using Newtonsoft.Json.Linq;

namespace OpenMined.Syft.Layer
{
    public class Log : Layer, LayerDefinition
    {

        [SerializeField] public string name = "log";

        public Log(SyftController controller)
        {
            init(this.name);

#pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public override FloatTensor Forward(FloatTensor input)
        {
            FloatTensor output = input.Log();
            activation = output.Id;

            return output;
        }
        
        public override int getParameterCount(){return 0;}

        public string GetLayerDefinition()
        {
            return JsonUtility.ToJson(this);
        }
        
        public override JToken GetConfig()
        {
            var config = new JObject
            {
                { "name", name }
            };

            return config;
        }
    }
}