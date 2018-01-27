using System;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using Newtonsoft.Json.Linq;
using OpenMined.Protobuf.Onnx;

namespace OpenMined.Syft.Layer
{
    public class Softmax : Layer, LayerDefinition
    {

        [SerializeField] public string name = "softmax";
        [SerializeField] private int dim = 0;
    
        public Softmax(SyftController controller, int dim)
        {
            init(this.name);

            this.dim = dim;
            
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public Softmax(SyftController controller, GraphProto graph)
        {
            init(this.name);

            this.dim = (int) graph.Node[0].Attribute[0].I;
            
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public override FloatTensor Forward(FloatTensor input)
        {
            FloatTensor output = input.Softmax(this.dim);
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
                { "name", name },
                { "dim", dim }
            };

            return config;
        }

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
                OpType = "Softmax",
                Attribute = { new AttributeProto{
                    Name = "axis",
                    Type = AttributeProto.Types.AttributeType.Int,
                    I = this.dim
                }}
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