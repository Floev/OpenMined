using System;
using UnityEngine;
using UnityEditor;
using NUnit.Framework;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using OpenMined.Network.Servers;
using UnityEditor.VersionControl;

namespace OpenMined.Tests.Editor.IntTensorTests
{
    [Category("IntTensorGPUTests")]
    public class IntTensorGPUTest
    {
        public SyftController ctrl;
        public ComputeShader shader;

        public void AssertEqualTensorsData(IntTensor t1, IntTensor t2, double delta = 0.0d)
        {

            int[] data1 = new int[t1.Size];
            t1.DataBuffer.GetData(data1);

            int[] data2 = new int[t2.Size];
            t2.DataBuffer.GetData(data2);

            Assert.AreEqual(t1.DataBuffer.count, t2.DataBuffer.count);
            Assert.AreEqual(t1.DataBuffer.stride, t2.DataBuffer.stride);
            Assert.AreNotEqual(t1.DataBuffer.GetNativeBufferPtr(), t2.DataBuffer.GetNativeBufferPtr());
            Assert.AreEqual(data1.Length, data2.Length);

            for (var i = 0; i < data1.Length; ++i)
            {
                //Debug.LogFormat("Asserting {0} equals {1} with accuracy {2} where diff is {3}", data1[i], data2[i], delta, data1[i] - data2[i]);
                Assert.AreEqual(data1[i], data2[i], delta);
            }
        }

        public void AssertApproximatelyEqualTensorsData(IntTensor t1, IntTensor t2)
        {
            AssertEqualTensorsData(t1, t2, .0001f);
        }

        public void AssertEqualTensorsData(OpenMined.Syft.Tensor.FloatTensor t1, OpenMined.Syft.Tensor.FloatTensor t2, double delta = 0.0d)
        {
            float[] data1 = new float[t1.Size];
            t1.DataBuffer.GetData(data1);
            float[] data2 = new float[t2.Size];
            t2.DataBuffer.GetData(data2);
            Assert.AreEqual(t1.DataBuffer.count, t2.DataBuffer.count);
            Assert.AreEqual(t1.DataBuffer.stride, t2.DataBuffer.stride);
            Assert.AreNotEqual(t1.DataBuffer.GetNativeBufferPtr(), t2.DataBuffer.GetNativeBufferPtr());
            for (var i = 0; i < data1.Length; ++i)
            {
                //Debug.LogFormat("Asserting {0} equals {1} with accuracy {2} where diff is {3}", data1[i], data2[i], delta, data1[i] - data2[i]);
                Assert.AreEqual(data1[i], data2[i], delta);
            }
        }

        public void AssertApproximatelyEqualTensorsData(OpenMined.Syft.Tensor.FloatTensor t1, OpenMined.Syft.Tensor.FloatTensor t2)
        {
            AssertEqualTensorsData(t1, t2, .0001f);
        }

        [OneTimeSetUp]
        public void Init()
        {
            //Init runs once before running test cases.
            ctrl = new SyftController(null);
            shader = Camera.main.GetComponents<SyftServer>()[0].Shader;
        }

        [OneTimeTearDown]
        public void CleanUp()
        {
            //CleanUp runs once after all test cases are finished.
        }

        [SetUp]
        public void SetUp()
        {
            //SetUp runs before all test cases
        }

        [TearDown]
        public void TearDown()
        {
            //SetUp runs after all test cases
        }

        /********************/
        /* Tests Start Here */
        /********************/

        [Test]
        public void Abs()
        {
            int[] shape1 = { 2, 5 };
            int[] data1 = { -1, -2, -3, -4, 5, 6, 7, 8, -999, 10 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            int[] expectedData1 = { 1, 2, 3, 4, 5, 6, 7, 8, 999, 10 };
            int[] shape2 = { 2, 5 };
            var expectedTensor1 = ctrl.intTensorFactory.Create(_data: expectedData1, _shape: shape2);
            expectedTensor1.Gpu(shader);

            var absTensor1 = tensor1.Abs();

            AssertEqualTensorsData(expectedTensor1, absTensor1);
        }

        [Test]
        public void Abs_()
        {
            int[] shape1 = { 2, 5 };
            int[] data1 = { -1, -2, -3, -4, 5, 6, 7, 8, -999, 10 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            int[] expectedData1 = { 1, 2, 3, 4, 5, 6, 7, 8, 999, 10 };
            int[] shape2 = { 2, 5 };
            var expectedTensor1 = ctrl.intTensorFactory.Create(_data: expectedData1, _shape: shape2);
            expectedTensor1.Gpu(shader);

            tensor1.Abs(inline: true);

            AssertEqualTensorsData(expectedTensor1, tensor1);
        }

        [Test]
        public void Add()
        {
            int[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            int[] data2 = {3, 2, 6, 9, 10, 1, 4, 8, 5, 7};
            int[] shape2 = {2, 5};
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);
            tensor2.Gpu(shader);

            int[] data3 = {4, 4, 9, 13, 15, 7, 11, 16, 14, 17};
            int[] shape3 = {2, 5};
            var expectedTensor = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);
            expectedTensor.Gpu(shader);

            var tensorSum = tensor1.Add(tensor2);

            AssertEqualTensorsData(expectedTensor, tensorSum);
        }

        [Test]
        public void Add_()
        {
            int[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            int[] data2 = {3, 2, 6, 9, 10, 1, 4, 8, 5, 7};
            int[] shape2 = {2, 5};
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);
            tensor2.Gpu(shader);

            int[] data3 = {4, 4, 9, 13, 15, 7, 11, 16, 14, 17};
            int[] shape3 = {2, 5};
            var expectedTensor = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);
            expectedTensor.Gpu(shader);

            tensor1.Add(tensor2, inline: true);

            AssertEqualTensorsData(expectedTensor, tensor1);
        }

        [Test]
        public void Eq()
        {
            int[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            int[] data2 = {3, 2, 6, 9, 1, 1, 4, 8, 5, 10};
            int[] shape2 = {2, 5};
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);
            tensor2.Gpu(shader);

            int[] expectedData = {0, 1, 0, 0, 0, 0, 0, 1, 0, 1};
            int[] ExpectedDataShape = {2, 5};
            var expectedTensor = ctrl.intTensorFactory.Create(_data: expectedData, _shape: ExpectedDataShape);
            expectedTensor.Gpu(shader);

            var resultTensor = tensor1.Eq(tensor2);

            AssertEqualTensorsData(expectedTensor, resultTensor);
        }

        [Test]
        public void Eq_()
        {
            int[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            int[] data2 = {3, 2, 6, 9, 1, 1, 4, 8, 5, 10};
            int[] shape2 = {2, 5};
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);
            tensor2.Gpu(shader);

            int[] expectedData = {0, 1, 0, 0, 0, 0, 0, 1, 0, 1};
            int[] ExpectedDataShape = {2, 5};
            var expectedTensor = ctrl.intTensorFactory.Create(_data: expectedData, _shape: ExpectedDataShape);
            expectedTensor.Gpu(shader);

            tensor1.Eq(tensor2, inline: true);

            AssertEqualTensorsData(expectedTensor, tensor1);
        }

        [Test]
        public void Sub()
        {
            int[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            int[] data2 = {3, 2, 6, 9, 10, 1, 4, 8, 5, 7};
            int[] shape2 = {2, 5};
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);
            tensor2.Gpu(shader);

            int[] data3 = {-2,  0, -3, -5, -5,  5,  3,  0,  4,  3};
            int[] shape3 = {2, 5};
            var expectedTensor = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);
            expectedTensor.Gpu(shader);

            var tensorDiff = tensor1.Sub(tensor2);

            AssertEqualTensorsData(expectedTensor, tensorDiff);
        }

        [Test]
        public void Sub_()
        {
            int[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            int[] data2 = {3, 2, 6, 9, 10, 1, 4, 8, 5, 7};
            int[] shape2 = {2, 5};
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);
            tensor2.Gpu(shader);

            int[] data3 = {-2,  0, -3, -5, -5,  5,  3,  0,  4,  3};
            int[] shape3 = {2, 5};
            var expectedTensor = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);
            expectedTensor.Gpu(shader);

            tensor1.Sub(tensor2, inline: true);

            AssertEqualTensorsData(expectedTensor, tensor1);
        }

		[Test]
		public void Neg()
		{
			int[] shape1 = { 2, 5 };
			int[] data1 = { -1, -2, -3, -4, 5, 6, 7, 8, -999, 10 };
			var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
			tensor1.Gpu(shader);

			int[] expectedData1 = { 1, 2, 3, 4, -5, -6, -7, -8, 999, -10 };
			int[] shape2 = { 2, 5 };
			var expectedTensor1 = ctrl.intTensorFactory.Create(_data: expectedData1, _shape: shape2);
			expectedTensor1.Gpu(shader);

			var actualTensorNeg1 = tensor1.Neg();

			AssertEqualTensorsData(expectedTensor1, actualTensorNeg1);
		}

		[Test]
		public void Neg_()
		{
			int[] shape1 = { 2, 5 };
			int[] data1 = { -1, -2, -3, -4, 5, 6, 7, 8, -999, 10 };
			var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
			tensor1.Gpu(shader);

			int[] expectedData1 = { 1, 2, 3, 4, -5, -6, -7, -8, 999, -10 };
			int[] shape2 = { 2, 5 };
			var expectedTensor1 = ctrl.intTensorFactory.Create(_data: expectedData1, _shape: shape2);
			expectedTensor1.Gpu(shader);

			tensor1.Neg(inline: true);

			AssertEqualTensorsData(expectedTensor1, tensor1);
		}

        [Test]
        public void Reciprocal()
        {
            int[] data1 = { 1, 2, 3, -1 };
            int[] shape1 = { 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            int[] data2 = { 1, 0, 0, -1 };
            int[] shape2 = { 4 };
            var expectedTensor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);
            expectedTensor.Gpu(shader);

            var reciprocalTensor = tensor1.Reciprocal();

            AssertEqualTensorsData(expectedTensor, reciprocalTensor);
        }

        [Test]
        public void Reciprocal_()
        {
            int[] data1 = { 1, 2, 3, -1 };
            int[] shape1 = { 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            int[] data2 = { 1, 0, 0, -1 };
            int[] shape2 = { 4 };
            var expectedTensor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);
            expectedTensor.Gpu(shader);

            tensor1.Reciprocal(inline: true);

            AssertEqualTensorsData(expectedTensor, tensor1);
        }

        [Test]
        public void Sin()
        {
            int[] data1 = { 15, 60, 90, 180 };
            int[] shape1 = { 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            float[] data2 = { 0.65028784f, -0.30481062f, 0.89399666f, -0.80115264f };
            int[] shape2 = { 4 };
            var expectedSinTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);
            expectedSinTensor.Gpu(shader);

            var actualSinTensor = tensor1.Sin();

            AssertApproximatelyEqualTensorsData(expectedSinTensor, actualSinTensor);
        }

        [Test]
        public void Cos()
        {
            int[] data1 = { 30, 60, 90, 180 };
            int[] shape1 = { 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            float[] data2 = { 0.1542515f, -0.952413f, -0.4480736f, -0.5984601f };
            int[] shape2 = { 4 };
            var expectedCosTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);
            expectedCosTensor.Gpu(shader);

            var actualCosTensor = tensor1.Cos();

            AssertApproximatelyEqualTensorsData(expectedCosTensor, actualCosTensor);
        }

        [Test]
        public void view()
        {
            int[] data1 = { 4, 4, 7, 7, 2, 2, 4, 8, 7, 8, 8, 0, 6, 2, 8, 9 };
            int[] shape1 = { 2, 2, 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            int[] data2 = { 4, 4, 7, 7, 2, 2, 4, 8, 7, 8, 8, 0, 6, 2, 8, 9 };
            int[] shape2 = { 8, 2 };
            var expectedIntTesnor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);
            expectedIntTesnor.Gpu(shader);

            var actualIntTensor = tensor1.View(shape2);
            AssertEqualTensorsData(expectedIntTesnor, actualIntTensor);
        }

        [Test]
        public void view_()
        {
            int[] data1 = { 4, 4, 7, 7, 2, 2, 4, 8, 7, 8, 8, 0, 6, 2, 8, 9 };
            int[] shape1 = { 2, 2, 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Gpu(shader);

            int[] data2 = { 4, 4, 7, 7, 2, 2, 4, 8, 7, 8, 8, 0, 6, 2, 8, 9 };
            int[] shape2 = { 8, 2 };
            var expectedIntTesnor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);
            expectedIntTesnor.Gpu(shader);

            tensor1.View(shape2, inline: true);
            AssertEqualTensorsData(expectedIntTesnor, tensor1);
        }
        /* closes class and namespace */
    }
}
