using System;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using OpenMined.Protobuf.Onnx;

namespace OpenMined.Syft.Layer
{
    public class Dropout : Layer, LayerDefinition
	{

        [SerializeField] public string name = "dropout";
        [SerializeField] private FloatTensor _mask_source;
        [SerializeField] private float rate;

		public Dropout(SyftController _controller, float _rate)
		{
			init(this.name);

			this.controller = _controller;
			this.rate = _rate;

			#pragma warning disable 420
			id = System.Threading.Interlocked.Increment(ref nCreated);
			controller.addModel(this);

		}

    public Dropout(SyftController _controller, GraphProto graph)
    {
      init(this.name);

      this.controller = _controller;
      this.rate = graph.Node[0].Attribute[0].F;

      #pragma warning disable 420
      id = System.Threading.Interlocked.Increment(ref nCreated);
      controller.addModel(this);

    }

		public override FloatTensor Forward(FloatTensor input)
		{
			if (_mask_source == null || input.Size != _mask_source.Size)
			{
				_mask_source = input.emptyTensorCopy(hook_graph:false);
				_mask_source.Fill(1 - rate, inline: true);
			};
			
			FloatTensor output = input.Mul(_mask_source.SampleMask());
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
            { "name", "dropout" },
            { "rate" , rate }
        };

        return config;
    }

    // See https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout
    public override GraphProto GetProto (int inputTensorId, SyftController ctrl)
    {
      FloatTensor input_tensor = ctrl.floatTensorFactory.Get(inputTensorId);
      this.Forward(input_tensor);

      NodeProto node = new NodeProto
      {
        Input = { inputTensorId.ToString() },
        Output = { activation.ToString(), _mask_source.Id.ToString() },
        Name = this.name,
        OpType = "Dropout",
        DocString = ""
      };
      node.Attribute.Add(new AttributeProto{
        Name = "ratio",
        Type = AttributeProto.Types.AttributeType.Float,
        F = this.rate
      });
      node.Attribute.Add(new AttributeProto{
        Name = "is_test",
        Type = AttributeProto.Types.AttributeType.Int,
        I = 1
      });

      ValueInfoProto input_info = input_tensor.GetValueInfoProto();

      GraphProto g =  new GraphProto
      {
        Name = Guid.NewGuid().ToString("N"),
        Node = { node },
        Initializer = {  },
        Input = { input_info },
        Output = { 
          ctrl.floatTensorFactory.Get(activation).GetValueInfoProto(),
          _mask_source.GetValueInfoProto()
        },
      };

      return g;
    }
  }
}
