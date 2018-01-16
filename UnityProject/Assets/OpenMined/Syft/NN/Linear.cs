using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer
{
	public class Linear: Layer
	{
		private readonly FloatTensor _weights;
		private FloatTensor _bias;
        private bool _biased;

        public Linear(SyftController _controller, int input, int output, string initializer = "Xavier", 
            bool biased = false, float[] weights = null, float[] bias = null)
		{
			init("linear");

			controller = _controller;
            _biased = biased || bias!=null;
						
			int[] weightShape = { input, output };
            if (weights == null)
            {
                weights = initializer == "Xavier" ? controller.RandomWeights(input * output, input) : controller.RandomWeights(input * output);
            };
			_weights = controller.floatTensorFactory.Create(_shape: weightShape, _data: weights, _autograd: true, _keepgrads: true);

            parameters.Add(_weights.Id);

            if (_biased)
            {
                int[] biasShape = { 1, output };
                _bias = controller.floatTensorFactory.Create(_data:bias,_shape: biasShape, _autograd: true);
                parameters.Add(_bias.Id);
            };
			
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
		
		public override int getParameterCount()
        {
            return _biased ? _weights.Size + _bias.Size : _weights.Size;
        }
	}
}