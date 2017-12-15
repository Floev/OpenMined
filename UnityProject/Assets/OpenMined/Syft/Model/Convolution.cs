//using JetBrains.Annotations;
using OpenMined.Network.Controllers;
//using OpenMined.Network.Utils;
using OpenMined.Syft.Tensor;
//using OpenMined.Syft.NN;
//using UnityEngine;
using System.Linq;

namespace OpenMined.Syft.Layer
{
    public class Conv2d : Model
    {
        private int _in_dim;
        private int _out_dim;
        private int[] _kernel_dims;
        private int[] _stride_dims;
        private int[] _padding_dims;
        private int[] _dilation_dims;
        private int _group;
        private bool _biased;
        //private readonly FloatTensor _kernel;
        private FloatTensor _kernel;
        private FloatTensor _bias;

        public Conv2d(SyftController _controller, int input, int output, int[] kernel,
            int[] stride = null, int[] padding = null, int[] dilation = null, int group = 1, bool bias = false)
        {
            init();
            if (stride == null)
                stride = new int[] { 1, 1 };
            if (kernel == null)
                kernel = new int[] { 1, 1 };
            if (padding == null)
                padding = new int[] { 1, 1 };
            if (dilation == null)
                dilation = new int[] { 1, 1 };

            this.controller = _controller;

            _in_dim = input;
            _out_dim = output;
            _kernel_dims = kernel;
            _stride_dims = stride;
            _padding_dims = padding;
            _dilation_dims = dilation;
            _group = group;
            _biased = bias;

            _kernel = new FloatTensor(controller, _shape: kernel, _data: controller.RandomWeights(kernel.Aggregate(1, (a, b) => a * b)), _autograd: true);
//            Debug.LogFormat("Init kernel: {0}", _kernel.Print());
            parameters.Add(_kernel.Id);

            if (_biased)
            {
                _bias = new FloatTensor(controller, _shape: new int[]{ output }, _autograd: true);
                _bias.Zero_();
                parameters.Add(_bias.Id);
            }

#pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public override FloatTensor Forward( FloatTensor input)
        {
            return NN.Functional.Conv2d(input, _kernel, _bias, _stride_dims, _padding_dims, _dilation_dims, _group);
        }

    }
}