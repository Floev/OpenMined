using System;
using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using Newtonsoft.Json.Linq;
using OpenMined.Protobuf.Onnx;

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

        public ReLU(SyftController controller, GraphProto graph)
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

        public override int getParameterCount() { return 0; }

        // Serialization
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

        // See https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu
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
                OpType = "Relu",
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