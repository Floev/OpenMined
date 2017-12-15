using JetBrains.Annotations;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer
{
	public class Linear: Model
	{
		private int _input;
		private int _output;

		private readonly FloatTensor _weights;
		private FloatTensor _bias;
		
		public Linear (SyftController _controller, int input, int output)
		{
			init();

			this.controller = _controller;
			
			_input = input;
			_output = output;
			
			int[] weightShape = { input, output };
			var weights = controller.RandomWeights(input * output);
			_weights = new FloatTensor(controller, _shape: weightShape, _data: weights, _autograd: true);

			// TODO: add bias when broadcast is available
			int[] biasShape = {output};
			_bias = new FloatTensor(controller, biasShape, _autograd: true);

			parameters.Add(_weights.Id);
			//parameters.Add(_bias.Id);
			
			#pragma warning disable 420
			id = System.Threading.Interlocked.Increment(ref nCreated);
			controller.addModel(this);
		}

		public override FloatTensor Forward(FloatTensor input)
		{
			return input.MM(_weights);
		}

	}
}