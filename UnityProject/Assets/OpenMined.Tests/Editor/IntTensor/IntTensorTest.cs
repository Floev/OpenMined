using System;
using NUnit.Framework;
using OpenMined.Network.Controllers;
using OpenMined.Protobuf;
using UnityEngine;
using Google.Protobuf;
using OpenMined.Protobuf.Onnx;

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
        public void Acos(){
            int[] data1 = { 1, 0, -1, 1, 1, 0 };
            int[] shape1 = { 6 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = { 0, 1.57079633f, 3.14159265f, 0, 0, 1.57079633f };
            int[] shape2 = { 6 };
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualTensorAcos = tensor1.Acos();

            for (int i = 0; i < actualTensorAcos.Size; i++)
            {
                Assert.AreEqual(actualTensorAcos[i], expectedTensor[i]);
            }
        }

        [Test]
        public void Add()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            var tensorSum = tensor1.Add(tensor2);

            for (int i = 0; i < tensorSum.Size; i++)
            {
                Assert.AreEqual(tensor1[i] + tensor2[i], tensorSum[i]);
            }
        }

        [Test]
        public void Add_()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            int[] data3 = { 4, 4, 9, 13, 15, 7, 11, 16, 14, 17 };
            int[] shape3 = { 2, 5 };
            var tensor3 = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);

            tensor1.Add(tensor2, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor3[i], tensor1[i]);
            }
        }

        [Test]
        public void AddScalar()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int scalar = 5;

            var tensorSum = tensor1.Add(scalar);

            for (int i = 0; i < tensorSum.Size; i++)
            {
                Assert.AreEqual(tensor1[i] + scalar, tensorSum[i]);
            }
        }

        [Test]
        public void AddScalar_()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int scalar = -5;

            int[] data2 = { -4, -3, -2, -1,  0,  1,  2,  3,  4,  5 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Add(scalar, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor2[i], tensor1[i]);
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

        [Test]
        public void Eq()
        {
            int[] data1 = { 1, 2, 3, 4 };
            int[] shape = { 2, 2 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape);

            int[] data2 = { 1, 2, 1, 2 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape);

            int[] expectedData = { 1, 1, 0, 0 };
            var expectedOutput = ctrl.intTensorFactory.Create(_data: expectedData, _shape: shape);

            var eqOutput = tensor1.Eq(tensor2);

            for (int i = 0; i < expectedOutput.Size; i++)
            {
                Assert.AreEqual(expectedOutput[i], eqOutput[i]);
            }
        }

        [Test]
        public void Eq_()
        {
            int[] data1 = { 1, 2, 3, 4 };
            int[] shape = { 2, 2 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape);

            int[] data2 = { 1, 2, 1, 2 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape);

            int[] expectedData = { 1, 1, 0, 0 };
            var expectedOutput = ctrl.intTensorFactory.Create(_data: expectedData, _shape: shape);

            tensor1.Eq(tensor2, inline:true);

            for (int i = 0; i < expectedOutput.Size; i++)
            {
                Assert.AreEqual(expectedOutput[i], tensor1[i]);
            }
        }

        [Test]
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
        public void Max_()
        {
            int[] data = { 4,0,6,-3,8,-2 };
            int[] compareData = { 1,-2,2,-3,0,-1 };
            int[] shape = { 1, 6 };
            var tensor = ctrl.intTensorFactory.Create(_data: data, _shape: shape);
            var compareTensor = ctrl.intTensorFactory.Create(_data: compareData, _shape: shape);


            int[] expectedData = { 4, 0, 6, -3, 8, -1 };
            var expectedOutput = ctrl.intTensorFactory.Create(_data: expectedData, _shape: shape);

            var maxOutput = tensor.Max(compareTensor, inline: true);

		for (int i = 0; i < expectedOutput.Size; i++)
  	          {
	                Assert.AreEqual(expectedOutput[i], maxOutput[i]);            
		}
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
        public void TopK()
        {
           
            // test #1 when largest True and sorted True
            int[] data1 = {2, 5, 3, 9, 5, 23, 19, 24, -9, -22};
            int[] shape1 = {10};
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            var actual1 = tensor1.TopK(4);
            int [] expected1 = {9,19,23,24};
            for (int i = 0; i < actual1.Size; i++)
            {
                Assert.AreEqual(expected1[i], actual1[i]);
            }
            // test #2 when largest False and sorted True
            int[] data2 = {2, 5, 3, 9, 5, 23, 19, 24, -9, -22,100,33};
            int[] shape2 = {12};
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);
            var actual2 = tensor2.TopK(5,largest: false,sorted: true);
            int [] expected2 = {-22,-9,2,3,5};
            for (int i = 0; i < actual2.Size; i++)
            {
                Assert.AreEqual(expected2[i], actual2[i]);
            }
            // test #3 when largest true, sorted false and dim 0
            int[] data3 = {21, 5, 3, 9, 5, 23};
            int[] shape3 = {2,3};
            var tensor3 = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);
            var actual3 = tensor3.TopK(1, dim:0, largest: true,sorted: false);
            int [] expected3 = {21,5,23};
            for (int i = 0; i < actual3.Size; i++)
            {
                Assert.AreEqual(expected3[i], actual3[i]);
            }

            // test #4 when largest false and sorted false and dim -1
            int[] data4 = { 5, 3, 9, 5 , 19, 24, -9, -22 };
            int[] shape4 = {2,2,2};
            var tensor4 = ctrl.intTensorFactory.Create(_data: data4, _shape: shape4);
            var actual4 = tensor4.TopK(1,dim:-1,largest: false,sorted: false);
            int[] expected4 = {3,5,19,-22};
           
            for (int i = 0; i < actual4.Size; i++)
            {
                Assert.AreEqual(expected4[i], actual4[i]);
            }
            // test #5 when k is largest than the tensor size
            Assert.That(() => tensor4.TopK(11,sorted: false),
                Throws.TypeOf<ArgumentException>());
           

            // test #5 when k is equal to the tensor size
            int[] data5 = {2, 5, 3, 9, 5, 23, 19, 24, -9, -22};
            int[] shape5 = {10};
            var tensor5 = ctrl.intTensorFactory.Create(_data: data5, _shape: shape5);
            var actual5 = tensor5.TopK(10,sorted: false);
            int [] expected5 = data5;
            for (int i = 0; i < actual5.Size; i++)
            {
                Assert.AreEqual(expected5[i], actual5[i]);
            }

            // TODO uncomment this when sort based on dimension implemented
            // test #5 when dim -2 and sorted true

            // int[] data6 = {2, 5, 3, 9, 21, 52, 31, 91, 6, 23, 33, 11};
            // int[] shape6 = {3,4};
            // var tensor6 = ctrl.intTensorFactory.Create(_data: data6, _shape: shape6);
            // var actual6 = tensor6.TopK(2,dim:-2,sorted: true);
            // int [] expected6 = {6,23 ,31,11,21,52,33,91,};
            // for (int i = 0; i < actual6.Size; i++)
            // {
            //     Assert.AreEqual(expected6[i], actual6[i]);
            // }

        }

        [Test]
        public void Exp()
        {
            int[] inputShape = { 2, 2 };
            int[] inputData = { -1, 1, 3, 34};
            var inputTensor = ctrl.intTensorFactory.Create(_data: inputData, _shape: inputShape);

            int[] outputShape = { 2, 2 };
            int[] expectedData = { 0, 2, 20, 2147483647};
            var expectedTensor = ctrl.intTensorFactory.Create(_data: expectedData, _shape: outputShape);

            var outputTensorExp = inputTensor.Exp();

            for (int i = 0; i < expectedTensor.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], outputTensorExp[i]);
            }
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

        [Test]
        public void Sinh()
        {
            int[] data1 = { -1, -2, 3, 4, 5, -6 };
            int[] shape1 = { 6 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = { -1.175201f, -3.62686f, 10.01787f, 27.28992f, 74.20321f, -201.7132f };
            int[] shape2 = { 6 };
            var expectedSinhTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualSinhTensor = tensor1.Sinh();

            for (int i = 0; i < actualSinhTensor.Size; i++)
            {
                Assert.AreEqual(expectedSinhTensor[i], actualSinhTensor[i], 1e-4);
            }
        }

        [Test]
        public void GetProto()
        {
            int[] data = {-1, 0, 1, int.MaxValue, int.MinValue};
            int[] shape = {5};
            Syft.Tensor.IntTensor t = ctrl.intTensorFactory.Create(_data: data, _shape: shape);

            TensorProto message = t.GetProto();
            byte[] messageAsByte = message.ToByteArray();
            TensorProto message2 = TensorProto.Parser.ParseFrom(messageAsByte);

            Assert.AreEqual(message, message2);
        }
        [Test]
        public void view()
        {
            int[] data1 = { 4, 4, 7, 7, 2, 2, 4, 8, 7, 8, 8, 0, 6, 2, 8, 9 };
            int[] shape1 = { 2, 2, 4 };
            var tesnor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 4, 4, 7, 7, 2, 2, 4, 8, 7, 8, 8, 0, 6, 2, 8, 9 };
            int[] shape2 = { 8, 2 };
            var expectedIntTesnor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            var actualIntTensor = tesnor1.View(shape2);
            for(int i = 0; i < actualIntTensor.Size; i++)
            {
                Assert.AreEqual(expectedIntTesnor[i], actualIntTensor[i]);
            }
        }

        [Test]
        public void view_()
        {
            int[] data1 = { 4, 4, 7, 7, 2, 2, 4, 8, 7, 8, 8, 0, 6, 2, 8, 9 };
            int[] shape1 = { 2, 2, 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 4, 4, 7, 7, 2, 2, 4, 8, 7, 8, 8, 0, 6, 2, 8, 9 };
            int[] shape2 = { 8, 2 };
            var expectedIntTesnor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.View(shape2, inline: true);
            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedIntTesnor[i], tensor1[i]);
            }
        }

        [Test]
        public void Unfold()
        {
            int[] input_data = {-1, 2, 3, 5, 0, 4, 6, 7, 10, 3, 2, -5};
            int[] input_shape = {3, 4};

            var input_tensor = ctrl.intTensorFactory.Create(_data: input_data, _shape: input_shape);

            // Test1
            int dim = 0;
            int step = 1;
            int size = 2;
            int[] expected_data = {-1, 2, 3, 5, 0, 4, 6, 7, 0, 4, 6, 7, 10, 3, 2, -5};
            int[] expected_shape = {2, 2, 4};
            var expected_output_tensor = ctrl.intTensorFactory.Create(_data: expected_data, _shape: expected_shape);

            var actual_output_tensor = input_tensor.Unfold(dim: dim, step: step, size: size);
            
            for (int i = 0; i<actual_output_tensor.Size; i++)
            {
                Assert.AreEqual(expected_output_tensor[i], actual_output_tensor[i]);
            }

            // Test2
            dim = 1;
            step = 1;
            size = 3;
            int[] expected_data2 = {-1, 2, 3, 0, 4, 6, 10, 3, 2, 2, 3, 5, 4, 6, 7, 3, 2, -5};
            int[] expected_shape2 = {2, 3, 3};
            var expected_output_tensor2 = ctrl.intTensorFactory.Create(_data: expected_data2, _shape: expected_shape2);

            actual_output_tensor = input_tensor.Unfold(dim: dim, step: step, size: size);
            
            for (int i = 0; i<actual_output_tensor.Size; i++)
            {
                Assert.AreEqual(expected_output_tensor2[i], actual_output_tensor[i]);
            }
        }

        /* closes class and namespace */
    }
}
