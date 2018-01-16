using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
	}
}