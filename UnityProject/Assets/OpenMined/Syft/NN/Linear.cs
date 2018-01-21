using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using OpenMined.Network.Servers;
using Newtonsoft.Json.Linq;

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

        public Linear (SyftController _controller, int input, int output, string initializer="Xavier",
            bool biased = false, float[] weights = null, float[] bias = null)
		{
            init(name);


			this.controller = _controller;
			
			_input = input;
			_output = output;

            _biased = biased || bias != null;

            int[] weightShape = { output, input };
            if (weights == null)
            {
                weights = initializer == "Xavier" ? controller.RandomWeights(input * output, input) : controller.RandomWeights(input * output);
            };
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

        public Linear (SyftController _controller, int input, int output, FloatTensor weights, FloatTensor bias, string initializer="Xavier")
        {
            init(this.name);

            this.controller = _controller;

            _input = input;
            _output = output;

            _weights = weights;
            _bias = bias;

            parameters.Add(_weights.Id);
            parameters.Add(_bias.Id);

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public override FloatTensor Forward(FloatTensor input)
		{
			
			FloatTensor output = input.MM(_weights);
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
                { "bias", _bias.GetConfig() },
                { "weights", _weights.GetConfig() },
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

	}
}

