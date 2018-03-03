using System;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using Newtonsoft.Json.Linq;
using OpenMined.Protobuf.Onnx;
using System.Linq;

namespace OpenMined.Syft.Layer
{
    public class Conv2d : Layer, LayerDefinition
    {
        [SerializeField] private FloatTensor _kernel;
        [SerializeField] private FloatTensor _bias;
        [SerializeField] private IntTensor _stride_dims;
        [SerializeField] private IntTensor _padding_dims;
        [SerializeField] private IntTensor _dilation_dims;
        [SerializeField] private IntTensor _group;
        public bool _biased;
        private bool _transposed;

        public Conv2d(SyftController _controller, int[] kernel,
            int[] stride = null, int[] padding = null, int[] dilation = null, int group = 1, bool biased = false, bool transposed = false,
            float[] kernelData = null, float[] biasData = null, string initializer = "Xavier")
        {
            init("conv2d");
            if (stride == null)
                stride = new int[] { 1, 1, 1, 1 };
            if (padding == null)
                padding = new int[] { 1, 1, 1, 1 };
            if (dilation == null)
                dilation = new int[] { 1, 1, 1, 1 };

            this.controller = _controller;

            _stride_dims = controller.intTensorFactory.Create(new int[] { 4 }, stride);
            _padding_dims = controller.intTensorFactory.Create(new int[] { 4 }, padding);
            _dilation_dims = controller.intTensorFactory.Create(new int[] { 4 }, dilation);
            _group = controller.intTensorFactory.Create(new int[] { 1 }, new int[] { group });
            _biased = biased || biasData != null;
            _transposed = transposed;
            float[] _kernelData = kernelData;
            if (_kernelData == null)
            {
                int initi = 0;
                if (initializer == "Xavier")
                    initi = kernel[1];
                _kernelData = controller.RandomWeights(kernel.Aggregate(1, (a, b) => a * b), initi);
            }
            else
            {
//                Debug.LogFormat("Trans for speed: {0}", String.Join(",", _kernelData));
                _kernelData = controller.floatTensorFactory.Create(_shape: kernel, _data: _kernelData).Transpose(0,1).Data;
//                Debug.LogFormat("into: {0}", String.Join(",", _kernelData));
            }
            var tmp = kernel[0];
            kernel[0] = kernel[1];
            kernel[1] = tmp;

            _kernel = controller.floatTensorFactory.Create(_shape: kernel, _data: _kernelData, _autograd: true, _keepgrads: true);
            parameters.Add(_kernel.Id);

            if (_biased)
            {
//                Debug.Log("Biased conv2d Layer");
                _bias = controller.floatTensorFactory.Create(_data: biasData, _shape: new int[] { 1, kernel[0], 1, 1 }, _autograd: true);
                parameters.Add(_bias.Id);
            };

#pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public override FloatTensor Forward(FloatTensor input)
        {
//            Debug.LogFormat("Apply conv2d to {0}", input.Print());
//            Debug.LogFormat("using {0}, {1}, {2}, {3}, {4}, {5}", _kernel.Print(), _stride_dims.Print(), _padding_dims.Print(), _dilation_dims.Print(), _group.Print(), _transposed);
            var output = input.Conv2d(_kernel, _stride_dims, _padding_dims, _dilation_dims, _group, _transposed);
//            Debug.LogFormat("Result {0}", output.Print());
            if (_biased)
            {
//                Debug.Log("Biased conv2d forward pass");
                //                Debug.LogFormat("Bias: {0}", _bias.Print());
                //                Debug.LogFormat("Expand from {0} to {1}", String.Join(",",_bias.Shape),String.Join(",", output.Shape));
                //                var exp_bias = _bias.Expand(output.Shape);
                //                Debug.LogFormat("Bias: {0}", exp_bias.Print());
                output = output.Add(_bias.Expand(output.Shape).Contiguous());
//                Debug.LogFormat("Result after bias {0}", output.Print());
            };

            activation = output.Id;

            return output;
        }

        public override int getParameterCount()
        {
            return _biased ? _kernel.Size + _bias.Size : _kernel.Size;
        }

        public string GetLayerDefinition()
        {
            return JsonUtility.ToJson(this);
        }
    }
}