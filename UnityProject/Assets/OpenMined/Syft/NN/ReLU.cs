using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using Newtonsoft.Json.Linq;

namespace OpenMined.Syft.Layer
{
    public class ReLU : Layer, LayerDefinition
    {

        [SerializeField] string name = "relu";

        public ReLU(SyftController controller)
        {
            init(this.name);

#pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public override FloatTensor Forward(FloatTensor input)
        {
            FloatTensor output = input.ReLU();
            activation = output.Id;

            return output;
        }

        public string GetLayerDefinition()
        {
            return JsonUtility.ToJson(this);
        }

        public override int getParameterCount() { return 0; }

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