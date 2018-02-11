using System;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using Newtonsoft.Json.Linq;
using OpenMined.Protobuf.Onnx;

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

        // See https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log
        public override GraphProto GetProto(int inputTensorId, SyftController ctrl)
        {
            FloatTensor input_tensor = ctrl.floatTensorFactory.Get(inputTensorId);
            if (activation != null)
            {
                this.Forward(input_tensor);
            }

            NodeProto node = new NodeProto
            {
                Input = { inputTensorId.ToString() },
                Output = { activation.ToString() },
                OpType = "Log",
            };

            ValueInfoProto input_info = input_tensor.GetValueInfoProto();

            GraphProto g =  new GraphProto
            {
                Name = Guid.NewGuid().ToString("N"),
                Node = { node },
                Initializer = {  },
                Input = { input_info },
                Output = { ctrl.floatTensorFactory.Get(activation).GetValueInfoProto() },
            };

            return g;            
        }
    }
}