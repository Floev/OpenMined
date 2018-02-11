using System;
using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using OpenMined.Network.Servers;
using Newtonsoft.Json.Linq;
using OpenMined.Protobuf.Onnx;
using Google.Protobuf.Collections;

namespace OpenMined.Syft.Layer
{
    
    public class Linear: Layer, LayerDefinition
	{
		private int _input;
		private int _output;

        [SerializeField] string name = "linear";
        [SerializeField] public FloatTensor _weights;
        [SerializeField] FloatTensor _bias;
        private bool _biased;
        private bool _fast;

        public Linear (SyftController _controller, int input, int output, string initializer="Xavier",
            bool biased = false, float[] weights = null, float[] bias = null, bool fast = true)
        {
            init(name);

			this.controller = _controller;

            _input = fast ? output : input;
            _output = fast ? input : output;
            _fast = fast;

            _biased = biased || bias != null;

            int[] weightShape = { _input, _output };
            if (weights == null)
            {
                weights = initializer == "Xavier" ? controller.RandomWeights(input * output, input) : controller.RandomWeights(input * output);
            }

            if (_fast)
            {
                var new_weights = new float[weights.Length];
                for (var idx = 0; idx < weights.Length; idx++)
                {
                    new_weights[(idx - (idx % output)) / output + input * (idx % output)] = weights[idx];
                }
                weights = new_weights;
            }

            _weights = controller.floatTensorFactory.Create(_shape: weightShape, _data: weights, _autograd: true, _keepgrads: true);

            parameters.Add(_weights.Id);

            if (_biased)
            {
                int[] biasShape = { 1, output };
                _bias = controller.floatTensorFactory.Create(_data: bias, _shape: biasShape, _autograd: true);
                parameters.Add(_bias.Id);
            };

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

		// Overloading the constructor to load from an ONNX proto
		public Linear (SyftController _controller, GraphProto graph)
		{
            init(this.name);

			this.controller = _controller;

            _weights = ONNXTools.BuildFloatTensor(graph.Initializer[0], this.controller, autograd: true, keepgrads: true);
			AttributeProto transB = ONNXTools.FindAttribute(graph.Node[0], "transB");
			if (transB != null && transB.I == 1)
			{
				_weights = _weights.Transpose();
			}
            parameters.Add(_weights.Id);
            _input = _weights.Shape[0];
            _output = _weights.Shape[1];

      
            _bias = ONNXTools.BuildFloatTensor(graph.Initializer[1], this.controller, autograd: true, keepgrads: true);
            _biased = true;  
            parameters.Add(_bias.Id);
      
			#pragma warning disable 420
			id = System.Threading.Interlocked.Increment(ref nCreated);
			controller.addModel(this);
        }

        public override FloatTensor Forward(FloatTensor input)
        {
            FloatTensor output;
            if (_fast) { output = input.MMT(_weights); }
            else { output = input.MM(_weights); };

            if (_biased)
            {
                output = output.Add(_bias.Expand(output.Shape).Contiguous());
            };
            activation = output.Id;
		
			return output;
		}

        public string GetLayerDefinition()
        {
            return JsonUtility.ToJson(this);
        }

        public override int getParameterCount()
        {
            return _biased ? _weights.Size + _bias.Size : _weights.Size;
        }

	  public override JToken GetConfig()
    {
		  var config = new JObject
			{
			    { "name", "linear" },
				{ "trainable", true },
				{ "dtype", "float32" }, 
				{ "output", _output },
                { "input", _input },
                { "bias", _bias?.GetConfig() },
                { "weights", _weights?.GetConfig() },
				{ "activation", "linear" },
				{ "use_bias", true },
				{
				    "kernel_initializer", new JObject
					{
					    { "class_name", "VarianceScaling" },
						{ 
						    "config", new JObject
						  	{
							    { "scale", 1.0 },
							  	{ "mode", "fan_avg" },
							  	{ "distribution", "uniform" },
							  	{ "seed", null }
						  	}
						}
					}
				},
				{ 
				    "bias_initializer", new JObject
					{
		          	    { "class_name", "Zeros"},
		          		{ "config", new JObject() }
		          	}
		        },
				{ "kernel_regularizer", null },
		        { "bias_regularizer", null },
		        { "activity_regularizer", null },
		        { "kernel_constraint", null },
		        { "bias_constraint", null }
				};

			return config;
		}

		// See https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
		public override GraphProto GetProto (int inputTensorId, SyftController ctrl)
		{
			FloatTensor input_tensor = ctrl.floatTensorFactory.Get(inputTensorId);
			this.Forward(input_tensor);

			NodeProto node = new NodeProto
			{
				Input = { inputTensorId.ToString(), _weights.Id.ToString() },
				Output = { activation.ToString() },
				Name = this.name,
				OpType = "Gemm",
				DocString = ""
			};
      if (_biased)
      {
        node.Input.Add(_bias.Id.ToString());
      }
      
			node.Attribute.Add(new AttributeProto{
				Name = "alpha",
				Type = AttributeProto.Types.AttributeType.Float,
				F = 1.0f
			});
			node.Attribute.Add(new AttributeProto{
				Name = "beta",
				Type = AttributeProto.Types.AttributeType.Float,
				F = 1.0f
			});
			node.Attribute.Add(new AttributeProto{
				Name = "broadcast",
				Type = AttributeProto.Types.AttributeType.Int,
				I = 1
			});

			TensorProto w_init = _weights.GetProto();

			ValueInfoProto input_info = input_tensor.GetValueInfoProto();
			ValueInfoProto w_info = _weights.GetValueInfoProto();

			GraphProto g =  new GraphProto
			{
        Name = Guid.NewGuid().ToString("N"),
				Node = { node },
				Initializer = { w_init },
				Input = { input_info, w_info },
				Output = { ctrl.floatTensorFactory.Get(activation).GetValueInfoProto() },
			};

      if (_biased)
      {
        TensorProto b_init = _bias.GetProto();
        ValueInfoProto b_info = _bias.GetValueInfoProto();
        g.Initializer.Add(b_init);
        g.Input.Add(b_info);
      }
      else
      {
        // The Gemm schema, must have 3 inputs (must have a bias)
        float[] tmpData = new float[1] {0};
        int[] tmpDims = new int[1] {1};
        FloatTensor tmpBias = ctrl.floatTensorFactory.Create(_data: tmpData, _shape: tmpDims, _autograd: false, _keepgrads: false);
        g.Initializer.Add(tmpBias.GetProto());
        g.Input.Add(tmpBias.GetValueInfoProto());
        g.Node[0].Input.Add(tmpBias.Id.ToString());
      }

			return g;
		}
	}
}