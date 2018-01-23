using System;
using NUnit.Framework;
using OpenMined.Network.Controllers;

namespace OpenMined.Tests.Tensor.IntTensor
{
    [Category("CPUTest")]
    public class CPUTest
    {
        private SyftController ctrl;

        [OneTimeSetUp]
        public void Init()
        {
            //Init runs once before running test cases.
            ctrl = new SyftController(null);
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

            int[] expectedData1 = { 1, 2, 3, 4, 5, 6, 7, 8, 999, 10 };
            int[] shape2 = { 2, 5 };
            var expectedTensor1 = ctrl.intTensorFactory.Create(_data: expectedData1, _shape: shape2);

            var actualTensorAbs1 = tensor1.Abs();

            for (int i = 0; i < actualTensorAbs1.Size; i++)
            {
                Assert.AreEqual(expectedTensor1[i], actualTensorAbs1[i]);
            }
        }

        [Test]
        public void Abs_()
        {
            int[] shape1 = { 2, 5 };
            int[] data1 = { -1, -2, -3, -4, 5, 6, 7, 8, -999, 10 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] expectedData1 = { 1, 2, 3, 4, 5, 6, 7, 8, 999, 10 };
            int[] shape2 = { 2, 5 };
            var expectedTensor1 = ctrl.intTensorFactory.Create(_data: expectedData1, _shape: shape2);

            var actualTensorAbs1 = tensor1.Abs(inline: true);

            for (int i = 0; i < actualTensorAbs1.Size; i++)
            {
                Assert.AreEqual(expectedTensor1[i], actualTensorAbs1[i]);
            }
        }

        [Test]
        public void Add()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var tensorSum = tensor1.Add(tensor2);

            for (int i = 0; i < tensorSum.Size; i++)
            {
                Assert.AreEqual(tensor1[i] + tensor2[i], tensorSum[i]);
            }
        }

        [Test]
        public void Add_()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            float[] data3 = { 4, 4, 9, 13, 15, 7, 11, 16, 14, 17 };
            int[] shape3 = { 2, 5 };
            var tensor3 = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);

            tensor1.Add(tensor2, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor3[i], tensor1[i]);
            }
        }

		[Test]
		public void Neg()
		{
			int[] shape1 = { 2, 5 };
			int[] data1 = { -1, -2, -3, -4, 5, 6, 7, 8, -999, 10 };
			var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

			int[] expectedData1 = { 1, 2, 3, 4, -5, -6, -7, -8, 999, -10 };
			int[] shape2 = { 2, 5 };
			var expectedTensor1 = ctrl.intTensorFactory.Create(_data: expectedData1, _shape: shape2);

			var actualTensorNeg1 = tensor1.Neg();

			for (int i = 0; i < actualTensorNeg1.Size; i++)
			{
				Assert.AreEqual(expectedTensor1[i], actualTensorNeg1[i]);
			}
		}

		[Test]
		public void Neg_()
		{
			int[] shape1 = { 2, 5 };
			int[] data1 = { -1, -2, -3, -4, 5, 6, 7, 8, -999, 10 };
			var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

			int[] expectedData1 = { 1, 2, 3, 4, -5, -6, -7, -8, 999, -10 };
			int[] shape2 = { 2, 5 };
			var expectedTensor1 = ctrl.intTensorFactory.Create(_data: expectedData1, _shape: shape2);

			tensor1.Neg(inline: true);

			for (int i = 0; i < tensor1.Size; i++)
			{
				Assert.AreEqual(expectedTensor1[i], tensor1[i]);
			}
		}

        [Test]
        public void Reciprocal()
        {
            int[] data1 = {1, 2, 3, -1};
            int[] shape1 = { 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = {1, 0, 0, -1};
            int[] shape2 = { 4 };
            var expectedTensor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            var actualTensor = tensor1.Reciprocal();

            for (int i = 0; i < expectedTensor.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], actualTensor[i]);
            }
        }

        [Test]
        public void Reciprocal_()
        {
            int[] data1 = { 1, 2, 3, -1 };
            int[] shape1 = { 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 1, 0, 0, -1 };
            int[] shape2 = { 4 };
            var expectedTensor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Reciprocal(inline: true);

            for (int i = 0; i < expectedTensor.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], tensor1[i]);
            }
        }

        public void Equal()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            var tensor3 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] differentShapedData = { 0, 0 };
            int[] differentShape = { 1, 2 };
            var differentShapedTensor = ctrl.intTensorFactory.Create(_data: differentShapedData, _shape: differentShape);

            Assert.False(tensor1.Equal(differentShapedTensor));
            Assert.False(tensor1.Equal(tensor2));
            Assert.True(tensor1.Equal(tensor3));
        }

        [Test]
        public void PowElem()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 5, 4, 3, 2, 1, 1, 2, 3, 4, 5 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            int[] data3 = { 1, 16, 27, 16, 5, 1, 4, 27, 256, 3125 };
            int[] shape3 = { 2, 5 };
            var tensor3 = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);

            var result = tensor1.Pow(tensor2);

            for (int i = 0; i < result.Size; i++)
            {
                Assert.AreEqual(tensor3[i], result[i]);
            }
        }

        [Test]
        public void PowElem_()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 5, 4, 3, 2, 1, 1, 2, 3, 4, 5 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            int[] data3 = { 1, 16, 27, 16, 5, 1, 4, 27, 256, 3125 };
            int[] shape3 = { 2, 5 };
            var tensor3 = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);

            tensor1.Pow(tensor2, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor3[i], tensor1[i]);
            }
        }

        [Test]
        public void PowScalar()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            var result = tensor1.Pow(2);

            for (int i = 0; i < result.Size; i++)
            {
                Assert.AreEqual(tensor2[i], result[i]);
            }
        }

        [Test]
        public void PowScalar_()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Pow(2, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor2[i], tensor1[i]);
            }
        }

        [Test]
        public void Sqrt()
        {
            int[] data1 = {1, 4, 9, 16};
            int[] shape1 = {4};

            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            var result = tensor1.Sqrt();

            int[] data2 = {1, 2, 3, 4};
            int[] shape2 = {4};
            var expectedTensor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            for (int i = 0; i < tensor1.Data.Length; i++)
            {
                Assert.AreEqual(expectedTensor[i], result[i], 1e-3);
            }
        }

        [Test]
        public void Sub()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            var tensorDiff = tensor1.Sub(tensor2);

            for (int i = 0; i < tensorDiff.Size; i++)
            {
                Assert.AreEqual(tensor1[i] - tensor2[i], tensorDiff[i]);
            }
        }

        [Test]
        public void Sub_()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            int[] data3 = { -2,  0, -3, -5, -5,  5,  3,  0,  4,  3 };
            int[] shape3 = { 2, 5 };
            var tensor3 = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);

            tensor1.Sub(tensor2, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor3[i], tensor1[i]);
            }
        }

        [Test]
        public void SubScalar()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int scalar = 5;

            var tensorDiff = tensor1.Sub(scalar);

            for (int i = 0; i < tensorDiff.Size; i++)
            {
                Assert.AreEqual(tensor1[i] - scalar, tensorDiff[i]);
            }
        }

        [Test]
        public void SubScalar_()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int scalar = 5;

            int[] data2 = { -4, -3, -2, -1,  0,  1,  2,  3,  4,  5 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Sub(scalar, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor2[i], tensor1[i]);
            }
        }

        [Test]
        public void Sign()
        {
            int[] data1 = {-1,2,3,-5,6,-10};
            int[] shape1 = {2,3};
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = {-1,1,1,-1,1,-1};
            int[] shape2 = {2,3};
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            var tensor3 = tensor1.Sign(inline: false);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor2[i], tensor3[i]);
            }
        }

        [Test]
        public void Tan()
        {
            float[] data1 = {30, 20, 40, 50};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {-6.4053312f, 2.23716094f, -1.11721493f, -0.27190061f};
            int[] shape2 = {4};
            var expectedTanTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualTanTensor = tensor1.Tan();

            for (int i = 0; i < actualTanTensor.Size; i++)
            {
                Assert.AreEqual(expectedTanTensor[i], actualTanTensor[i]);
            }
        }

        [Test]
        public void Trace()
        {
            // test #1
            int[] data1 = {2, 2, 3, 4};
            int[] shape1 = {2, 2};
            var tensor = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            int actual = tensor.Trace();
            int expected = 6;

            Assert.AreEqual(expected, actual);

            // test #2
            int[] data3 = {1, 2, 3};
            int[] shape3 = {3};
            var non2DTensor = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);
            Assert.That(() => non2DTensor.Trace(),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void Lt()
        {
            int[] data1 = { 1, 2, 3, 4 };
            int[] shape = { 2, 2 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape);

            int[] data2 = { 2, 2, 1, 2 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape);

            int[] expectedData = { 1, 0, 0, 0 };
            var expectedOutput = ctrl.intTensorFactory.Create(_data: expectedData, _shape: shape);

            var ltOutput = tensor1.Lt(tensor2);

            for (int i = 0; i < expectedOutput.Size; i++)
            {
                Assert.AreEqual(expectedOutput[i], ltOutput[i]);
            }
        }

        [Test]
        public void Lt_()
        {
            int[] data1 = { 1, 2, 3, 4 };
            int[] shape = { 2, 2 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape);

            int[] data2 = { 2, 2, 1, 2 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape);

            int[] expectedData = { 1, 0, 0, 0 };
            var expectedOutput = ctrl.intTensorFactory.Create(_data: expectedData, _shape: shape);

            tensor1.Lt(tensor2, inline:true);

            for (int i = 0; i < expectedOutput.Size; i++)
            {
                Assert.AreEqual(expectedOutput[i], tensor1[i]);
            }
        }

        [Test]
        public void Sin()
        {
            int[] data1 = { 15, 60, 90, 180 };
            int[] shape1 = { 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = { 0.65028784f, -0.30481062f, 0.89399666f, -0.80115264f };
            int[] shape2 = { 4 };
            var expectedSinTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualSinTensor = tensor1.Sin();

            for (int i = 0; i < actualSinTensor.Size; i++)
            {
                Assert.AreEqual(expectedSinTensor[i], actualSinTensor[i]);
            }
        }

        [Test]
        public void Cos()
        {
            int[] data1 = { 30, 60, 90, 180 };
            int[] shape1 = { 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = { 0.1542515f, -0.952413f, -0.4480736f, -0.5984601f };
            int[] shape2 = { 4 };
            var expectedCosTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualCosTensor = tensor1.Cos();

            for (int i = 0; i < actualCosTensor.Size; i++)
            {
                Assert.AreEqual(expectedCosTensor[i], actualCosTensor[i], 0.00001f);
            }
        }

        /* closes class and namespace */
    }
}
