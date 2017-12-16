using System;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using NUnit.Framework;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Layer;
using OpenMined.Syft.Layer.Loss;
using OpenMined.Syft.Model;
using OpenMined.Syft.Tensor;
using UnityEngine;

namespace OpenMined.Tests.Editor.Model
{
    [Category("ModelCPUTests")]
    public class ModelTest
    {
        public SyftController ctrl;

        [OneTimeSetUp]
        public void Init()
        {
            //Init runs once before running test cases.
            ctrl = new SyftController(null);
        }

        [Test]
        public void Conv2d()
        {
            float[] inputData = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            int[] inputShape = new int[] { 1, 1, 3, 3 };
            int[] kernel = new int[] { 2, 2 };
            float[] kernelData = new float[] { 1, 1, 1, 1 };
            float[] outputData = new float[] { 12, 16, 24, 28 };
            int[] outputShape = new int[] { 1, 1, 2, 2 };

            var input = new Syft.Tensor.FloatTensor(_controller: ctrl, _data: inputData, _shape: inputShape);
            var target = new Syft.Tensor.FloatTensor(_controller: ctrl, _data: outputData, _shape: outputShape);

            var conv = new Syft.Layer.Conv2d(ctrl, 2, 2, kernel, kernelData: kernelData);
            var output = conv.Forward(input);
            for (int i = 0; i < target.Size; i++)
            {
                Assert.AreEqual(output[i], target[i]);
            }

            float[] inputData2 = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
            int[] inputShape2 = new int[] { 1, 2, 3, 3 };
            var input2 = new Syft.Tensor.FloatTensor(_controller: ctrl, _data: inputData2, _shape: inputShape2);
            float[] outputData2 = new float[] { 12, 16, 24, 28, 48, 52, 60, 64 };
            int[] outputShape2 = new int[] { 1, 2, 2, 2 };

            var target2 = new Syft.Tensor.FloatTensor(_controller: ctrl, _data: outputData2, _shape: outputShape2);

            var conv2 = new Syft.Layer.Conv2d(ctrl, 2, 2, kernel, kernelData: kernelData);
            var output2 = conv2.Forward(input2);
            for (int i = 0; i < target2.Size; i++)
            {
                Assert.AreEqual(output2[i], target2[i]);
            }

            float[] inputData3 = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            int[] inputShape3 = new int[] { 1, 1, 3, 3 };
            int[] kernel3 = new int[] { 3, 2 };
            float[] kernelData3 = new float[] { 1, 1, 1, 1, 1, 1 };
            var input3 = new Syft.Tensor.FloatTensor(_controller: ctrl, _data: inputData3, _shape: inputShape3);
            float[] outputData3 = new float[] { 27, 33 };
            int[] outputShape3 = new int[] { 1, 1, 1, 2 };

            var target3 = new Syft.Tensor.FloatTensor(_controller: ctrl, _data: outputData3, _shape: outputShape3);

            var conv3 = new Syft.Layer.Conv2d(ctrl, 2, 2, kernel3,kernelData:kernelData3);
            var output3 = conv3.Forward(input3);
            for (int i = 0; i < output3.Size; i++)
            {
                Assert.AreEqual(output3[i], target3[i]);
            }
        }

        [Test]
        public void Conv2dTransposed()
        {
            float[] inputData = new float[] { 1, 2, 3, 4 };
            int[] inputShape = new int[] { 1, 1, 2, 2 };
            int[] kernel = new int[] { 2, 2 };
            float[] kernelData = new float[] { 1, 1, 1, 1 };
            float[] outputData = new float[] { 1, 3, 2, 4, 10, 6, 3, 7, 4 };
            int[] outputShape = new int[] { 1, 1, 3, 3 };

            var input = new Syft.Tensor.FloatTensor(_controller: ctrl, _data: inputData, _shape: inputShape);
            var target = new Syft.Tensor.FloatTensor(_controller: ctrl, _data: outputData, _shape: outputShape);
            var conv = new Syft.Layer.Conv2d(ctrl, 2, 2, kernel, transposed: true, kernelData: kernelData);
            var output = conv.Forward(input);
            for (int i = 0; i < output.Size; i++)
            {
                Assert.AreEqual(output[i], target[i]);
            }

            int[] kernel2 = new int[] { 3, 1 };
            float[] kernelData2 = new float[] { 1, 1, 1 };
            float[] outputData2 = new float[] { 1, 2, 4, 6, 4, 6, 3, 4 };
            int[] outputShape2 = new int[] { 1, 1, 4, 2 };

            var target2 = new Syft.Tensor.FloatTensor(_controller: ctrl, _data: outputData2, _shape: outputShape2);
            var conv2 = new Syft.Layer.Conv2d(ctrl, 2, 2, kernel2, transposed: true, kernelData: kernelData2);
            var output2 = conv2.Forward(input);
            for (int i = 0; i < output2.Size; i++)
            {
                Assert.AreEqual(output2[i], target2[i]);
            }
        }
    }
}
