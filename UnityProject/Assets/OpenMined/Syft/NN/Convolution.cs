using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using System.Linq;
using System;

namespace OpenMined.Syft.Layer
{
    public class Conv2d : Layer
    {
        private FloatTensor _kernel;
        private FloatTensor _bias;
        private IntTensor _stride_dims;
        private IntTensor _padding_dims;
        private IntTensor _dilation_dims;
        private IntTensor _group;
        private bool _biased;
        private bool _transposed;

        public Conv2d(SyftController _controller, int input, int output, int[] kernel,
            int[] stride = null, int[] padding = null, int[] dilation = null, int group = 1, bool biased = false, bool transposed = false,
            float[] kernelData = null, float[] biasData = null, string initializer = "Xavier")
        {
            init("conv2d");
            if (stride == null)
                stride = new int[] { 1, 1 };
            if (kernel == null)
                kernel = new int[] { 1, 1 };
            if (padding == null)
                padding = new int[] { 1, 1 };
            if (dilation == null)
                dilation = new int[] { 1, 1 };

            this.controller = _controller;

            _stride_dims = controller.intTensorFactory.Create(new int[] { 2 }, stride);
            _padding_dims = controller.intTensorFactory.Create(new int[] { 2 }, padding);
            _dilation_dims = controller.intTensorFactory.Create(new int[] { 2 }, dilation);
            _group = controller.intTensorFactory.Create(new int[] { 1 }, new int[] { group });
            _biased = biased || biasData != null;
            if (_biased)
            {
                throw new NotImplementedException("Conv with bias requires enhancement of ExpandNewDimension");
            };
            _transposed = transposed;
            int[] kernelShape = new int[] { output * group / input, kernel[0], kernel[1] };
            float[] _kernelData = kernelData;
            if (_kernelData == null)
            {
                int initi = 0;
                if (initializer == "Xavier")
                    initi = input * kernelShape[0] * kernelShape[1];
                _kernelData = controller.RandomWeights(kernelShape.Aggregate(1, (a, b) => a * b), initi);
            };

            _kernel = controller.floatTensorFactory.Create(_shape: kernelShape, _data: _kernelData, _autograd: true,_keepgrads:true);
            parameters.Add(_kernel.Id);

            if (_biased)
            {
                _bias = controller.floatTensorFactory.Create(_data: biasData, _shape: new int[] { 1, output }, _autograd: true);
                //before shape was { output }
                parameters.Add(_bias.Id);
            };

            parameters.Add(_stride_dims.Id);
            parameters.Add(_padding_dims.Id);
            parameters.Add(_dilation_dims.Id);
            parameters.Add(_group.Id);

#pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public override FloatTensor Forward( FloatTensor input)
        {
//            Debug.LogFormat("Apply conv2d to {0}", input.Print());
//            Debug.LogFormat("using {0}, {1}, {2}, {3}, {4}, {5}", _kernel.Print(), _stride_dims.Print(), _padding_dims.Print(), _dilation_dims.Print(), _group.Print(), _transposed);
            var output = input.Conv2d(_kernel, _stride_dims, _padding_dims, _dilation_dims, _group, _transposed);
            if (_biased)
            {
                output = output.Add(_bias.Expand(output.Shape).Contiguous());
            };

            activation = output.Id;

            return output;
        }

        public override int getParameterCount()
        {
            return _biased ? _kernel.Size + _bias.Size : _kernel.Size;
        }
    }
}