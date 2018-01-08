using JetBrains.Annotations;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using OpenMined.Syft.Tensor;
//using OpenMined.Syft.NN;
using UnityEngine;
using System.Linq;

namespace OpenMined.Syft.Layer
{
    public class Conv2d : Layer
    {
        //private int _in_dim;
        private int _out_dim;
        private int[] _kernel_dims;
        private FloatTensor _stride_dims;
        private FloatTensor _padding_dims;
        private FloatTensor _dilation_dims;
        private FloatTensor _group;
        private bool _biased;
        private bool _transposed;
        //private readonly FloatTensor _kernel;
        private FloatTensor _kernel;
        private FloatTensor _bias;

        public Conv2d(SyftController _controller, int input, int output, int[] kernel,
            int[] stride = null, int[] padding = null, int[] dilation = null, int group = 1, bool bias = false, bool transposed = false,
            float[] kernelData = null)
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

            //_in_dim = input;
            _out_dim = output;
            _kernel_dims = kernel;
            _stride_dims = controller.floatTensorFactory.Create(new int[] { 2 }, stride.Select(x => (float)x).ToArray());
            _padding_dims = controller.floatTensorFactory.Create(new int[] { 2 }, padding.Select(x => (float)x).ToArray());
            _dilation_dims = controller.floatTensorFactory.Create(new int[] { 2 }, dilation.Select(x => (float)x).ToArray());
            _group = controller.floatTensorFactory.Create(new int[] { 1 }, new float[] { group });
            _biased = bias;
            _transposed = transposed;
            int[] kernelShape = new int[] { output * group / input, kernel[0], kernel[1] };
            float[] _kernelData = kernelData;
            if (_kernelData == null) _kernelData = controller.RandomWeights(kernelShape.Aggregate(1, (a, b) => a * b));

            _kernel = controller.floatTensorFactory.Create(_shape: kernelShape, _data: _kernelData, _autograd: true);
            parameters.Add(_kernel.Id);
_biased = false;
            if (_biased)
            {
                Debug.Log("Nooooooooo");
                _bias = controller.floatTensorFactory.Create(_shape: new int[] { output }, _autograd: true);
            }
            else
            {
                _bias = controller.floatTensorFactory.Create(_shape: new int[] { output });
                _bias.Zero_();
            }
            parameters.Add(_bias.Id);
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
            /*
            Debug.LogFormat("Forwarding {0}:", string.Join(",", input.Shape));
            Debug.LogFormat("_kernel shape {0}:", string.Join(",",_kernel.Shape));
            Debug.LogFormat("_kernel {0}:", _kernel.Print());
            Debug.LogFormat("_bias shape {0}:", string.Join(",", _bias.Shape));
            Debug.LogFormat("_bias {0}:", _bias.Print());
            Debug.LogFormat("_stride_dims {0}:", _stride_dims.Print());
            Debug.LogFormat("_padding_dims {0}:", _padding_dims.Print());
            Debug.LogFormat("_dilation_dims {0}:", _dilation_dims.Print());
            Debug.LogFormat("_group {0}:", _group.Print());
            Debug.LogFormat("_transposed {0}:", _transposed);
            */
            return input.Conv2d(_kernel, _bias, _stride_dims, _padding_dims, _dilation_dims, _group,_transposed);
        }

        public override int getParameterCount() { return _kernel.Size + _bias.Size; }
    }
}