using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OpenMined.Syft.Tensor;
using UnityEngine;

namespace OpenMined.Syft.NN
{
    public static class Functional
    {
        // TODO: Softmax will run on GPU, when below OPS have a GPU implementation!
        // TODO: Improve the implementation!!!
        public static FloatTensor Softmax(FloatTensor input, int dim = -1)
        {
            if (!input.IsContiguous())
                throw new NotImplementedException(
                    "Softmax Gradient does not support non-contiguous tensors at the moment!");

            //TODO: GPU support
            var gpu = false;
            if (input.DataOnGpu)
            {
                input.Cpu();
                gpu = true;
            }

            var _dim = (dim == -1) ? input.Shape.Length - 1 : dim;

            var outerSize = 1;
            var innerSize = 1;
            var dimSize = input.Shape[_dim];

            for (var i = 0; i < _dim; ++i)
                outerSize *= input.Shape[i];

            for (var i = _dim + 1; i < input.Shape.Length; ++i)
                innerSize *= input.Shape[i];

            var dimStride = innerSize;
            var outerStride = dimSize * dimStride;

            var output = input.Copy();


            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = (outerSize * innerSize) * (workerId + 1) / nCpu;
                for (var i = (outerSize * innerSize) * workerId / nCpu; i < max; i++)
                {
                    int outerIdx = i / innerSize;
                    int innerIdx = i % innerSize;

                    // works for contiguous!!
                    var index = outerIdx * outerStride + innerIdx;

                    var inputMax = float.MinValue;
                    for (var d = 0; d < dimSize; d++)
                    {
                        if (output.Data[d * dimStride] >= inputMax)
                            inputMax = output.Data[d * dimStride];
                    }

                    float sum = 0;
                    for (var d = 0; d < dimSize; d++)
                    {
                        var z = (float) Math.Exp(output.Data[index + d * dimStride] - inputMax);
                        output.Data[index + d * dimStride] = z;
                        sum += z;
                    }

                    float invSum = 1 / sum;
                    for (var d = 0; d < dimSize; d++)
                    {
                        output.Data[index + d * dimStride] = output.Data[index + d * dimStride] * invSum;
                    }
                }
            });

            if (gpu)
            {
                output.Gpu(input.Shader);
            }

            output = input.HookAutograd(ref output, "softmax-" + _dim.ToString(), false);

            return output;
        }

        public static FloatTensor SoftmaxGradient(FloatTensor output, FloatTensor gradOutput, int dim)
        {
            if (!output.IsContiguous() || !gradOutput.IsContiguous())
                throw new NotImplementedException(
                    "Softmax Gradient does not support non-contiguous tensors at the moment!");
            var outerSize = 1;
            var innerSize = 1;
            var dimSize = output.Shape[dim];

            for (var i = 0; i < dim; ++i)
                outerSize *= output.Shape[i];

            for (var i = dim + 1; i < output.Shape.Length; ++i)
                innerSize *= output.Shape[i];

            var dimStride = innerSize;
            var outerStride = dimSize * dimStride;

            var gradInput = output.emptyTensorCopy();

            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = (outerSize * innerSize) * (workerId + 1) / nCpu;
                for (var i = (outerSize * innerSize) * workerId / nCpu; i < max; i++)
                {
                    int outerIdx = i / innerSize;
                    int innerIdx = i % innerSize;

                    // works for contiguous!!
                    var index = outerIdx * outerStride + innerIdx;

                    float sum = 0;
                    for (var d = 0; d < dimSize; d++)
                        sum += output.Data[index + d * dimStride] * gradOutput.Data[index + d * dimStride];

                    for (var d = 0; d < dimSize; d++)
                        gradInput.Data[index + d * dimStride] =
                            output.Data[index + d * dimStride] * (gradOutput.Data[index + d * dimStride] - sum);
                }
            });

            gradInput.Autograd = false;

            return gradInput;
        }
    }
}