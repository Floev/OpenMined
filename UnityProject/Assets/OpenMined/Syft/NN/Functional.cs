using System;
using System.Linq;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.NN
{
    public static class Functional
    {

        public static FloatTensor Conv2d(FloatTensor input, FloatTensor kernel, FloatTensor bias, int[] stride, int[] padding, int[] dilation, int group)
        {
            int[] viewShape = { input.Shape[0], input.Shape[1] * input.Shape[2] };
            int[] kernelShape = { kernel.Shape[0] * kernel.Shape[1], 1 };
            /*
                        Debug.LogFormat("input: {0}", input.Print());
                        Debug.LogFormat("inputShape: {0}", string.Join(",",viewShape));
                        Debug.LogFormat("reshaped: {0}", input.View(viewShape).Print());
                        Debug.LogFormat("kernel: {0}", _kernel.Print());
                        Debug.LogFormat("kernelShape: {0}", string.Join(",", kernelShape));
            //            Debug.LogFormat("reshaped: {1}", _kernel.View(kernelShape).Print());
            */
            return input.View(viewShape).MM(kernel.View(kernelShape));
        }

        public static FloatTensor Softmax(FloatTensor input, int dim = -1)
        {
            
            // TODO -- GPU Support
            
            var copy = input.emptyTensorCopy();
            if (dim == -1)
            {
                dim = input.Strides.Length - 1;
            }

            input.ForEach(dim, (vals, offset, stride) =>
            {
                var sum = vals.Sum(d => (float) Math.Pow(Math.E, d));
                for (var v = 0; v < vals.Length; ++v)
                {
                    copy[offset + v * stride] = (float) Math.Pow(Math.E, input[offset + v * stride]) / sum;
                }
            });
			
            return copy;
        }
    }
}