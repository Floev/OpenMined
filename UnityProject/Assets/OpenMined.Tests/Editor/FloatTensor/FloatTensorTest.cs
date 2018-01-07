using System;
using NUnit.Framework;
using OpenMined.Network.Controllers;

namespace OpenMined.Tests.Editor.FloatTensor
{
    [Category("FloatTensorCPUTests")]
    public class FloatTensorCPUTest
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
            float[] data1 = {-1, 0, 1, float.MaxValue, float.MinValue};
            int[] shape1 = {5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 0, 1, float.MaxValue, -float.MinValue};
            int[] shape2 = {5};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var tensor2 = tensor1.Abs();

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], tensor2[i]);
            }
        }

        [Test]
        public void Abs_()
        {
            float[] data1 = {-1, 0, 1, float.MaxValue, float.MinValue};
            int[] shape1 = {5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 0, 1, float.MaxValue, -float.MinValue};
            int[] shape2 = {5};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Abs(inline: true);
            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], tensor1[i]);
            }
        }

        [Test]
        public void Acos()
        {
            float[] data1 = {0.4f, 0.5f, 0.3f, -0.1f};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1.15927948f, 1.04719755f, 1.26610367f, 1.67096375f};
            int[] shape2 = {4};
            var expectedAcosTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualAcosTensor = tensor1.Acos();

            for (int i = 0; i < actualAcosTensor.Size; i++)
            {
                Assert.AreEqual(expectedAcosTensor[i], actualAcosTensor[i]);
            }
        }

        [Test]
        public void Acos_()
        {
            float[] data1 = {0.4f, 0.5f, 0.3f, -0.1f};
            int[] shape1 = {4};
            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1.15927948f, 1.04719755f, 1.26610367f, 1.67096375f};
            int[] shape2 = {4};
            var expectedAcosTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor.Acos(inline: true);

            for (int i = 0; i < tensor.Size; i++)
            {
                Assert.AreEqual(expectedAcosTensor[i], tensor[i]);
            }
        }

        [Test]
        public void Add()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {3, 2, 6, 9, 10, 1, 4, 8, 5, 7};
            int[] shape2 = {2, 5};
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
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {3, 2, 6, 9, 10, 1, 4, 8, 5, 7};
            int[] shape2 = {2, 5};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            float[] data3 = {4, 4, 9, 13, 15, 7, 11, 16, 14, 17};
            int[] shape3 = {2, 5};
            var tensor3 = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);

            tensor1.Add(tensor2, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor3[i], tensor1[i]);
            }
        }

        [Test]
        public void AddMatrixMultiply()
        {
            float[] base1_data = new float[] {1, 2, 3, 4};
            int[] base1_shape = new int[] {2, 2};
            var base1 = ctrl.floatTensorFactory.Create(_data: base1_data, _shape: base1_shape);

            float[] base2_data = new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9};
            int[] base2_shape = new int[] {3, 3};
            var base2 = ctrl.floatTensorFactory.Create(_data: base2_data, _shape: base2_shape);

            float[] data = new float[] {1, 2, 3, 4, 5, 6};
            int[] tensor1_shape = new int[] {2, 3};
            int[] tensor2_shape = new int[] {3, 2};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data, _shape: tensor1_shape);
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data, _shape: tensor2_shape);

            base1.AddMatrixMultiply(tensor1, tensor2);
            base2.AddMatrixMultiply(tensor2, tensor1);

            for (int i = 0; i < base1_shape[0]; i++)
            {
                for (int j = 0; j < base1_shape[1]; j++)
                {
                    float mm_res = base1_data[i * base1_shape[1] + j];
                    for (int k = 0; k < tensor1_shape[1]; k++)
                    {
                        mm_res += tensor1[i, k] * tensor2[k, j];
                    }
                    Assert.AreEqual(base1[i, j], mm_res);
                }
            }

            for (int i = 0; i < base2_shape[0]; i++)
            {
                for (int j = 0; j < base2_shape[1]; j++)
                {
                    float mm_res = base2_data[i * base2_shape[1] + j];
                    for (int k = 0; k < tensor2_shape[1]; k++)
                    {
                        mm_res += tensor2[i, k] * tensor1[k, j];
                    }
                    Assert.AreEqual(base2[i, j], mm_res);
                }
            }
        }

        [Test]
        public void AddMatrixVectorProduct()
        {
            float[] baseData = new float[] {1, 2};
            int[] baseShape = new int[] {2};
            var baseVector = ctrl.floatTensorFactory.Create(_data: baseData, _shape: baseShape);

            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = new int[] {2, 2};
            var matrix = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = new float[] {5, 6};
            int[] shape2 = new int[] {2};
            var vector = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            baseVector.AddMatrixVectorProduct(matrix, vector);

            float[] expectedData = new float[] {18, 41};
            int[] expectedShape = new int[] {2};
            var expectedVector =
                ctrl.floatTensorFactory.Create(_data: expectedData, _shape: expectedShape);

            for (int i = 0; i < expectedVector.Size; i++)
            {
                Assert.AreEqual(expectedVector[i], baseVector[i]);
            }
        }

        [Test]
        public void AddScalar()
        {
            float[] data1 = {-1, 0, 0.1f, 1, float.MaxValue, float.MinValue};
            int[] shape1 = {3, 2};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float scalar = -100;

            var tensorSum = tensor1.Add(scalar);

            for (int i = 0; i < tensorSum.Size; i++)
            {
                Assert.AreEqual(tensor1[i] + scalar, tensorSum[i]);
            }
        }

        [Test]
        public void AddScalar_()
        {
            float[] data1 = {-1, 0, 1, float.MaxValue, float.MinValue};
            int[] shape1 = {5, 1};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {-101, -100, -99, float.MaxValue - 100, float.MinValue - 100};
            int[] shape2 = {5, 1};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            float scalar = -100;

            tensor1.Add(scalar, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor1[i], tensor2[i]);
            }
        }

        [Test]
        public void AddUnequalDimensions()
        {
            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4};
            int[] shape2 = {2, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Add(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void AddUnequalDimensions_()
        {
            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4};
            int[] shape2 = {2, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Add(tensor2, inline: true),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void AddUnequalShapes()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6};
            int[] shape1 = {2, 3};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6};
            int[] shape2 = {3, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Add(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void AddUnequalShapes_()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6};
            int[] shape1 = {2, 3};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6};
            int[] shape2 = {3, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Add(tensor2, inline: true),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void AddUnequalSizes()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape2 = {2, 6};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Add(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void AddUnequalSizes_()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape2 = {2, 6};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Add(tensor2, inline: true),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void Asin()
        {
            float[] data1 = {0.4f, 0.5f, 0.3f, -0.1f};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {0.41151685f, 0.52359878f, 0.30469265f, -0.10016742f};
            int[] shape2 = {4};
            var expectedAsinTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualAsinTensor = tensor1.Asin();

            for (int i = 0; i < actualAsinTensor.Size; i++)
            {
                Assert.AreEqual(expectedAsinTensor[i], actualAsinTensor[i]);
            }
        }

        [Test]
        public void Asin_()
        {
            float[] data1 = {0.4f, 0.5f, 0.3f, -0.1f};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {0.41151685f, 0.52359878f, 0.30469265f, -0.10016742f};
            int[] shape2 = {4};
            var expectedAsinTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Asin(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedAsinTensor[i], tensor1[i]);
            }
        }

        [Test]
        public void Atan()
        {
            float[] data1 = {30, 20, 40, 50};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1.53747533f, 1.52083793f, 1.54580153f, 1.55079899f};
            int[] shape2 = {4};
            var expectedAtanTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualAtanTensor = tensor1.Atan();

            for (int i = 0; i < actualAtanTensor.Size; i++)
            {
                Assert.AreEqual(expectedAtanTensor[i], actualAtanTensor[i]);
            }
        }

        [Test]
        public void Atan_()
        {
            float[] data1 = {30, 20, 40, 50};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1.53747533f, 1.52083793f, 1.54580153f, 1.55079899f};
            int[] shape2 = {4};
            var expectedAtanTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Atan(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedAtanTensor[i], tensor1[i]);
            }
        }

        [Test]
        public void Ceil()
        {
            float[] data1 = {5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f};
            int[] shape1 = {5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {6, -20, 9, 101, 101};
            int[] shape2 = {5};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var result = tensor1.Ceil();

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], result[i]);
            }
        }

        [Test]
        public void Ceil_()
        {
            float[] data1 = {5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f};
            int[] shape1 = {5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {6, -20, 9, 101, 101};
            int[] shape2 = {5};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Ceil(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], tensor1[i]);
            }
        }

        [Test]
        public void Clamp()
        {
            float[] data1 = {0, 1, 2, 3, 4, 5,6};
            int[] shape1 = {1, 7};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {2, 2, 2, 3, 4, 5, 5};
            int[] shape2 = {1, 7};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            float[] data3 = {2, 2, 2, 3, 4, 5, 6};
            int[] shape3 = {1, 7};
            var tensor3 = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);

            float min = 2;
            float max = 5;

            var tensorClampMinMax = tensor1.Clamp(min, max);

            for (int i = 0; i < tensorClampMinMax.Size; i++)
            {
                Assert.AreEqual(tensor2[i], tensorClampMinMax[i]);
            }

            var tensorClampMin = tensor1.Clamp(min);

            for (int i = 0; i < tensorClampMin.Size; i++)
            {
                Assert.AreEqual(tensor3[i], tensorClampMin[i]);
            }
        }

        [Test]
        public void Copy()
        {
            float[] array = {1, 2, 3, 4, 5};
            int[] shape = {5};

            var tensor = ctrl.floatTensorFactory.Create(_data: array, _shape: shape);
            var copy = tensor.Copy(tensor.Autograd);

            Assert.AreEqual(copy.Shape, tensor.Shape);
            Assert.AreEqual(copy.Data, tensor.Data);
            Assert.AreNotEqual(copy.Id, tensor.Id);
        }

        [Test]
        public void Cos()
        {
            float[] data1 = {0.4f, 0.5f, 0.3f, -0.1f};
            int[] shape1 = {4};
            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {0.92106099f, 0.87758256f, 0.95533649f, 0.99500417f};
            int[] shape2 = {4};
            var expectedCosTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualCosTensor = tensor.Cos();

            for (int i = 0; i < actualCosTensor.Size; i++)
            {
                Assert.AreEqual(expectedCosTensor[i], actualCosTensor[i]);
            }
        }

        [Test]
        public void Cos_()
        {
            float[] data1 = {0.4f, 0.5f, 0.3f, -0.1f};
            int[] shape1 = {4};
            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {0.92106099f, 0.87758256f, 0.95533649f, 0.99500417f};
            int[] shape2 = {4};
            var expectedCosTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor.Cos(inline: true);

            for (int i = 0; i < tensor.Size; i++)
            {
                Assert.AreEqual(expectedCosTensor[i], tensor[i]);
            }
        }


        [Test]
        public void Cosh()
        {
            float[] data1 = {0.4f, 0.5f, 0.3f, -0.1f};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1.08107237f, 1.12762597f, 1.04533851f, 1.00500417f};
            int[] shape2 = {4};
            var expectedCoshTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualCoshTensor = tensor1.Cosh();

            for (int i = 0; i < actualCoshTensor.Size; i++)
            {
                Assert.AreEqual(expectedCoshTensor[i], actualCoshTensor[i]);
            }
        }

        [Test]
        public void Cosh_()
        {
            float[] data1 = {0.4f, 0.5f, 0.3f, -0.1f};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1.08107237f, 1.12762597f, 1.04533851f, 1.00500417f};
            int[] shape2 = {4};
            var expectedCoshTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Cosh(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedCoshTensor[i], tensor1[i]);
            }
        }

        [Test]
        public void Create1DTensor()
        {
            float[] array = {1, 2, 3, 4, 5};
            int[] shape = {5};

            var tensor = ctrl.floatTensorFactory.Create(_data: array, _shape: shape);

            Assert.AreEqual(array.Length, tensor.Size);

            for (int i = 0; i < array.Length; i++)
            {
                Assert.AreEqual(array[i], tensor[i]);
            }
        }

        [Test]
        public void Create2DTensor()
        {
            float[,] array = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}};

            float[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape = {2, 5};

            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            Assert.AreEqual(array.GetLength(0), tensor.Shape[0]);
            Assert.AreEqual(array.GetLength(1), tensor.Shape[1]);

            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    Assert.AreEqual(array[i, j], tensor[i, j]);
                }
            }
        }

        [Test]
        public void Create3DTensor()
        {
            float[,,] array = {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}};

            float[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape = {2, 3, 2};

            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            Assert.AreEqual(array.GetLength(0), tensor.Shape[0]);
            Assert.AreEqual(array.GetLength(1), tensor.Shape[1]);
            Assert.AreEqual(array.GetLength(2), tensor.Shape[2]);

            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    for (int k = 0; k < array.GetLength(2); k++)
                    {
                        Assert.AreEqual(array[i, j, k], tensor[i, j, k]);
                    }
                }
            }
        }

        [Test]
        public void DivisionElementwise()
        {
            float[] data1 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape1 = {2, 4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape2 = {2, 4};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var tensorMult = tensor1.Div(tensor2);

            for (int i = 0; i < tensorMult.Size; i++)
            {
                Assert.AreEqual(tensorMult[i], tensor1[i] / tensor2[i]);
            }
        }

        [Test]
        public void DivisionElementwise_()
        {
            float[] data1 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape1 = {2, 4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 1, 1, (float) Double.NaN, 1, 1, 1, 1};
            int[] shape2 = {2, 4};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Div(tensor1, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor2[i], tensor1[i]);
            }
        }

        [Test]
        public void DivisionElementwiseUnequalDimensions()
        {
            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4};
            int[] shape2 = {2, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Div(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void DivisionElementwiseUnequalDimensions_()
        {
            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4};
            int[] shape2 = {2, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Div(tensor2, inline: true),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void DivisionElementwiseUnequalShapes()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6};
            int[] shape1 = {2, 3};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6};
            int[] shape2 = {3, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Div(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void DivisionElementwiseUnequalShapes_()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6};
            int[] shape1 = {2, 3};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6};
            int[] shape2 = {3, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Div(tensor2, inline: true),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void DivisionElementwiseUnequalSizes()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape2 = {2, 6};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Div(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void DivisionElementwiseUnequalSizes_()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape2 = {2, 6};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Div(tensor2, inline: true),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void DivisionScalar()
        {
            float[] data1 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape1 = {2, 4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            // Test division by 0
            float scalar = 0;
            var result = tensor1.Div(scalar);

            float[] data2 =
            {
                float.MinValue / scalar, -10 / scalar, -1.5f / scalar, 0 / scalar, 1.5f / scalar, 10 / scalar,
                20 / scalar, float.MaxValue / scalar
            };
            int[] shape2 = {2, 4};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], result[i]);
            }
            // Test division
            float[] data3 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape3 = {2, 4};
            var tensor3 = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);

            scalar = 99;
            tensor3.Div(scalar, inline: true);

            float[] data4 =
            {
                float.MinValue / scalar, -10 / scalar, -1.5f / scalar, 0 / scalar, 1.5f / scalar, 10 / scalar,
                20 / scalar, float.MaxValue / scalar
            };
            int[] shape4 = {2, 4};
            var expectedTensor2 = ctrl.floatTensorFactory.Create(_data: data4, _shape: shape4);

            for (int i = 0; i < tensor3.Size; i++)
            {
                Assert.AreEqual(expectedTensor2[i], tensor3[i]);
            }
        }

        [Test]
        public void DivisionScalar_()
        {
            float[] data1 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape1 = {2, 4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            // Test division by 0
            float scalar = 0;
            var result = tensor1.Div(scalar);

            float[] data2 =
            {
                float.MinValue / scalar, -10 / scalar, -1.5f / scalar, 0 / scalar, 1.5f / scalar, 10 / scalar,
                20 / scalar, float.MaxValue / scalar
            };
            int[] shape2 = {2, 4};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], result[i]);
            }
            // Test division
            float[] data3 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape3 = {2, 4};
            var tensor3 = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);

            scalar = 99;
            var result2 = tensor3.Div(scalar);

            float[] data4 =
            {
                float.MinValue / scalar, -10 / scalar, -1.5f / scalar, 0 / scalar, 1.5f / scalar, 10 / scalar,
                20 / scalar, float.MaxValue / scalar
            };
            int[] shape4 = {2, 4};
            var expectedTensor2 = ctrl.floatTensorFactory.Create(_data: data4, _shape: shape4);

            for (int i = 0; i < tensor3.Size; i++)
            {
                Assert.AreEqual(expectedTensor2[i], result2[i]);
            }
        }

        [Test]
        public void Exp()
        {
            float[] data1 = {0, 1, 2, 5};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1f, 2.71828183f, 7.3890561f, 148.4131591f};
            int[] shape2 = {4};
            var expectedExpTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualExpTensor = tensor1.Exp();

            for (int i = 0; i < actualExpTensor.Size; i++)
            {
                Assert.AreEqual(expectedExpTensor[i], actualExpTensor[i], 1e-3);
            }
        }

        [Test]
        public void Exp_()
        {
            float[] data1 = {0, 1, 2, 5};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1f, 2.71828183f, 7.3890561f, 148.4131591f};
            int[] shape2 = {4};
            var expectedExpTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Exp(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedExpTensor[i], tensor1[i], 1e-3);
            }
        }

        [Test]
        public void Expand()
        {
            float[] data = {1, 2, 3, 4};
            int[] shape = {4, 1};

            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            int[] newShape = {4, 4};

            var expandedTensor = tensor.Expand(newShape);

            Assert.AreEqual(4, expandedTensor.Shape[0]);
            Assert.AreEqual(4, expandedTensor.Shape[1]);
            Assert.AreEqual(0, expandedTensor.Strides[1]);
        }

        [Test]
        public void ExpandNewDimension()
        {
            float[] data = {1, 2, 3, 4};
            int[] shape = {4, 1};

            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            int[] newShape = {4, 4, 4};

            var expandedTensor = tensor.Expand(newShape);

            foreach (int s in expandedTensor.Shape)
            {
                Assert.AreEqual(4, s);
            }

            Assert.AreEqual(0, expandedTensor.Strides[0]);
            Assert.AreEqual(0, expandedTensor.Strides[2]);
        }


        [Test]
        public void Floor()
        {
            float[] data1 = {5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f};
            int[] shape1 = {5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {5, -21, 9, 100, 100};
            int[] shape2 = {5};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var result = tensor1.Floor(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], result[i]);
            }
        }

        [Test]
        public void Floor_()
        {
            float[] data1 = {5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f};
            int[] shape1 = {5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {5, -21, 9, 100, 100};
            int[] shape2 = {5};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Floor(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], tensor1[i]);
            }
        }

        [Test]
        public void GetIndexGetIndeces()
        {
            float[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape = {3, 2, 2};
            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            int[] idxs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
            var idxarrays = new int[][]
            {
                new int[] {0, 0, 0},
                new int[] {0, 1, 0},
                new int[] {2, 0, 1}
            };

            foreach (var i in idxs)
                Assert.AreEqual(i, tensor.GetIndex(tensor.GetIndices(i)));
            foreach (var i in idxarrays)
                Assert.AreEqual(i, tensor.GetIndices(tensor.GetIndex(i)));
        }

        [Test]
        public void IsContiguous()
        {
            float[] data = new float[] {1, 2, 3, 4};
            int[] shape = new int[] {4, 1};
            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);
            Assert.AreEqual(tensor.IsContiguous(), true);

            var newShape = new int[] {4, 4};

            var expandedTensor = tensor.Expand(newShape);
            Assert.AreEqual(expandedTensor.IsContiguous(), false);
        }

        [Test]
        public void Log1p()
        {
            float[] data1 = {-0.4183f, 0.3722f, -0.3091f, 0.4149f, 0.5857f};
            int[] shape1 = {5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            float[] data2 = {-0.54180f, 0.31642f, -0.36976f, 0.34706f, 0.46103f};
            int[] shape2 = {5};
            var tensorLog1p = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var result = tensor1.Log1p();

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensorLog1p[i], result[i], 1e-5);
            }
        }

        [Test]
        public void Log1p_()
        {
            float[] data1 = {-0.4183f, 0.3722f, -0.3091f, 0.4149f, 0.5857f};
            int[] shape1 = {5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            float[] data2 = {-0.54180f, 0.31642f, -0.36976f, 0.34706f, 0.46103f};
            int[] shape2 = {5};
            var tensorLog1p = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Log1p(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensorLog1p[i], tensor1[i], 1e-5);
            }
        }

        [Test]
        public void Max()
        {
            float[] data = new float[]
            {
                1, 2, 3,
                4, 5, 6
            };

            int[] shape = new int[] {2, 3};
            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            Console.WriteLine("tensor {0}", string.Join(", ", tensor.Data));

            Syft.Tensor.FloatTensor result = tensor.Max();

            Console.WriteLine("tensor.Max() {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 1);
            Assert.AreEqual(result.Size, 1);
            Assert.AreEqual(result[0], 6.0);

            result = tensor.Max(0);

            Console.WriteLine("tensor.Max(0) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 3);
            Assert.AreEqual(result.Size, 3);
            Assert.AreEqual(result[0], 4.0);
            Assert.AreEqual(result[1], 5.0);
            Assert.AreEqual(result[2], 6.0);

            result = tensor.Max(1);

            Console.WriteLine("tensor.Max(1) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 2);
            Assert.AreEqual(result.Size, 2);
            Assert.AreEqual(result[0], 3.0);
            Assert.AreEqual(result[1], 6.0);

            /*result = tensor.Max(0, true);

            Console.WriteLine("tensor.Max(0, true) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 2);
            Assert.AreEqual(result.Shape[0], 1);
            Assert.AreEqual(result.Shape[1], 3);
            Assert.AreEqual(result.Size, 3);
            Assert.AreEqual(result[0], 4.0);
            Assert.AreEqual(result[1], 5.0);
            Assert.AreEqual(result[2], 6.0);*/
        }

        [Test]
        public void Mean()
        {
            float[] data = new float[]
            {
                1, 2, 3,
                4, 5, 6
            };

            int[] shape = new int[] {2, 3};
            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            Console.WriteLine("tensor {0}", string.Join(", ", tensor.Data));

            Syft.Tensor.FloatTensor result = tensor.Mean();

            Console.WriteLine("tensor.Mean() {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 1);
            Assert.AreEqual(result.Size, 1);
            Assert.AreEqual(result[0], 21.0 / 6.0);

            result = tensor.Mean(0);

            Console.WriteLine("tensor.Mean(0) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 3);
            Assert.AreEqual(result.Size, 3);
            Assert.AreEqual(result[0], 5.0 / 2.0);
            Assert.AreEqual(result[1], 7.0 / 2.0);
            Assert.AreEqual(result[2], 9.0 / 2.0);

            result = tensor.Mean(1);

            Console.WriteLine("tensor.Mean(1) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 2);
            Assert.AreEqual(result.Size, 2);
            Assert.AreEqual(result[0], 6.0 / 3.0);
            Assert.AreEqual(result[1], 15.0 / 3.0);

            /*result = tensor.Mean(0, true);

            Console.WriteLine("tensor.Mean(0, true) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 2);
            Assert.AreEqual(result.Shape[0], 1);
            Assert.AreEqual(result.Shape[1], 3);
            Assert.AreEqual(result.Size, 3);
            Assert.AreEqual(result[0], 5.0 / 2.0);
            Assert.AreEqual(result[1], 7.0 / 2.0);
            Assert.AreEqual(result[2], 9.0 / 2.0);*/
        }

        [Test]
        public void Min()
        {
            float[] data = new float[]
            {
                1, 2, 3,
                4, 5, 6
            };

            int[] shape = new int[] {2, 3};
            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            Console.WriteLine("tensor {0}", string.Join(", ", tensor.Data));

            Syft.Tensor.FloatTensor result = tensor.Min();

            Console.WriteLine("tensor.Min() {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 1);
            Assert.AreEqual(result.Size, 1);
            Assert.AreEqual(result[0], 1.0);

            result = tensor.Min(0);

            Console.WriteLine("tensor.Min(0) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 3);
            Assert.AreEqual(result.Size, 3);
            Assert.AreEqual(result[0], 1.0);
            Assert.AreEqual(result[1], 2.0);
            Assert.AreEqual(result[2], 3.0);

            result = tensor.Min(1);

            Console.WriteLine("tensor.Min(1) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 2);
            Assert.AreEqual(result.Size, 2);
            Assert.AreEqual(result[0], 1.0);
            Assert.AreEqual(result[1], 4.0);

            /*result = tensor.Min(0, true);

            Console.WriteLine("tensor.Min(0, true) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 2);
            Assert.AreEqual(result.Shape[0], 1);
            Assert.AreEqual(result.Shape[1], 3);
            Assert.AreEqual(result.Size, 3);
            Assert.AreEqual(result[0], 1.0);
            Assert.AreEqual(result[1], 2.0);
            Assert.AreEqual(result[2], 3.0);*/
        }

        [Test]
        public void MultiplicationElementwise()
        {
            float[] data1 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape1 = {2, 4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape2 = {2, 4};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var tensorMult = tensor1.Mul(tensor2);

            for (int i = 0; i < tensorMult.Size; i++)
            {
                Assert.AreEqual(tensorMult[i], tensor1[i] * tensor2[i]);
            }
        }

        [Test]
        public void MultiplicationElementwise_()
        {
            float[] data1 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape1 = {2, 4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape2 = {2, 4};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            float[] data3 = {float.PositiveInfinity, 100, 2.25f, 0, 2.25f, 100, 400, float.PositiveInfinity};
            int[] shape3 = {2, 4};
            var tensorMult = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);

            tensor1.Mul(tensor2, inline: true);

            for (int i = 0; i < tensorMult.Size; i++)
            {
                Assert.AreEqual(tensorMult[i], tensor1[i]);
            }
        }

        [Test]
        public void MutiplicationlElementwiseUnequalDimensions()
        {
            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4};
            int[] shape2 = {2, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Mul(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void MultiplicationElementwisenUnequalDimensions_()
        {
            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4};
            int[] shape2 = {2, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Mul(tensor2, inline: true),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void MultiplicationElementwiseUnequalShapes()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6};
            int[] shape1 = {2, 3};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6};
            int[] shape2 = {3, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);
            Assert.That(() => tensor1.Mul(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void MultiplicationElementwiseUnequalShapes_()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6};
            int[] shape1 = {2, 3};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6};
            int[] shape2 = {3, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);
            Assert.That(() => tensor1.Mul(tensor2, inline: true),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void MultiplicationElementwiseUnequalSizes()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape2 = {2, 6};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Mul(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void MultiplicationElementwiseUnequalSizes_()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape2 = {2, 6};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Mul(tensor2, inline: true),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void MultiplicationScalar()
        {
            float[] data1 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape1 = {2, 4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            // Test multiplication by 0
            float scalar = 0;
            var result = tensor1.Mul(scalar);
            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor2[i] * scalar, result[i]);
            }

            // Test multiplication by positive
            tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            scalar = 99;
            result = tensor1.Mul(scalar);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor2[i] * scalar, result[i]);
            }

            // Test multiplication by negative
            tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            scalar = -99;
            result = tensor1.Mul(scalar);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor2[i] * scalar, result[i]);
            }

            // Test multiplication by decimal
            tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            scalar = 0.000001f;
            result = tensor1.Mul(scalar);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor2[i] * scalar, result[i]);
            }
        }

        [Test]
        public void Neg()
        {
            float[] data1 = {-1, 0, 1, float.MaxValue, float.MinValue};
            int[] shape1 = {5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 0, -1, -float.MaxValue, -float.MinValue};
            int[] shape2 = {5};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var result = tensor1.Neg();

            for (int i = 0; i < result.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], result[i]);
            }
        }

        [Test]
        public void Neg_()
        {
            float[] data1 = {-1, 0, 1, float.MaxValue, float.MinValue};
            int[] shape1 = {5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 0, -1, (-1 * float.MaxValue), (-1 * float.MinValue)};
            int[] shape2 = {5};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Neg(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], tensor1[i]);
            }
        }

        [Test]
        public void Prod()
        {
            float[] data = new float[]
            {
                1, 2, 3,
                4, 5, 6
            };

            int[] shape = new int[] {2, 3};
            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            Console.WriteLine("tensor {0}", string.Join(", ", tensor.Data));

            Syft.Tensor.FloatTensor result = tensor.Prod();

            Console.WriteLine("tensor.Prod() {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 1);
            Assert.AreEqual(result.Size, 1);
            Assert.AreEqual(result[0], 720.0);

            result = tensor.Prod(0);

            Console.WriteLine("tensor.Prod(0) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 3);
            Assert.AreEqual(result.Size, 3);
            Assert.AreEqual(result[0], 4.0);
            Assert.AreEqual(result[1], 10.0);
            Assert.AreEqual(result[2], 18.0);

            result = tensor.Prod(1);

            Console.WriteLine("tensor.Prod(1) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 2);
            Assert.AreEqual(result.Size, 2);
            Assert.AreEqual(result[0], 6.0);
            Assert.AreEqual(result[1], 120.0);

         /*   result = tensor.Prod(0, true);

            Console.WriteLine("tensor.Prod(0, true) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 2);
            Assert.AreEqual(result.Shape[0], 1);
            Assert.AreEqual(result.Shape[1], 3);
            Assert.AreEqual(result.Size, 3);
            Assert.AreEqual(result[0], 4.0);
            Assert.AreEqual(result[1], 10.0);
            Assert.AreEqual(result[2], 18.0);*/
        }

        [Test]
        public void Reciprocal()
        {
            float[] data1 = {1f, 2f, 3f, 4f};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1f, 0.5f, 0.33333333f, 0.25f};
            int[] shape2 = {4};
            var expectedReciprocalTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualReciprocalTensor = tensor1.Reciprocal();

            for (int i = 0; i < actualReciprocalTensor.Size; i++)
            {
                Assert.AreEqual(expectedReciprocalTensor[i], actualReciprocalTensor[i], 1e-3);
            }
        }

        [Test]
        public void Reciprocal_()
        {
            float[] data1 = {1f, 2f, 3f, 4f};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1f, 0.5f, 0.33333333f, 0.25f};
            int[] shape2 = {4};
            var expectedReciprocalTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Reciprocal(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedReciprocalTensor[i], tensor1[i], 1e-3);
            }
        }


        [Test]
        public void RemainderElem()
        {
            float[] data = {-10, -5, -3.5f, 4.5f, 10, 20};
            float[] data_divisor = {2, 3, 1.5f, 0.5f, 7, 9};
            float[] data_expected = {0, -2, -0.5f, 0, 3, 2};
            int[] shape = {2, 3};

            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);
            var divisor = ctrl.floatTensorFactory.Create(_data: data_divisor, _shape: shape);
            var expected = ctrl.floatTensorFactory.Create(_data: data_expected, _shape: shape);

            var result = tensor.Remainder(divisor);

            for (int i = 0; i < tensor.Size; i++)
            {
                Assert.AreEqual(result[i], expected[i], 5e-7);
            }
        }

        [Test]
        public void RemainderElem_()
        {
            float[] data = {-10, -5, -3.5f, 4.5f, 10, 20};
            float[] data_divisor = {2, 3, 1.5f, 0.5f, 7, 9};
            float[] data_expected = {0, -2, -0.5f, 0, 3, 2};
            int[] shape = {2, 3};

            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);
            var divisor = ctrl.floatTensorFactory.Create(_data: data_divisor, _shape: shape);
            var expected = ctrl.floatTensorFactory.Create(_data: data_expected, _shape: shape);

            tensor.Remainder(divisor, true);

            for (int i = 0; i < tensor.Size; i++)
            {
                Assert.AreEqual(tensor[i], expected[i], 5e-7);
            }
        }

        [Test]
        public void RemainderScalar()
        {
            float[] data = {-10, -5, -3.5f, 4.5f, 10, 20};
            int[] shape = {2, 3};
            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            float[] res_mod2 = {0, -1, -1.5f, 0.5f, 0, 0};
            float[] res_mod3 = {-1, -2, -0.5f, 1.5f, 1, 2};
            float[] res_mod1p2 = {-0.4f, -0.2f, -1.1f, 0.9f, 0.4f, 0.8f};

            var out_mod2 = tensor.Remainder(2);
            var out_mod3 = tensor.Remainder(3);
            var out_mod1p2 = tensor.Remainder(1.2f);

            for (int i = 0; i < tensor.Size; i++)
            {
                Assert.AreEqual(out_mod2[i], res_mod2[i], 5e-7);
                Assert.AreEqual(out_mod3[i], res_mod3[i], 5e-7);
                Assert.AreEqual(out_mod1p2[i], res_mod1p2[i], 5e-6);
            }
        }

        [Test]
        public void RemainderScalar_()
        {
            float[] data = {-10, -5, -3.5f, 4.5f, 10, 20};
            int[] shape = {2, 3};
            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            float[] res_mod3 = {-1, -2, -0.5f, 1.5f, 1, 2};

            tensor.Remainder(3, inline: true);
            for (int i = 0; i < tensor.Size; i++)
            {
                Assert.AreEqual(tensor[i], res_mod3[i], 5e-7);
            }
        }

        [Test]
        public void Round()
        {
            float[] data1 = {5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f};
            int[] shape1 = {5};
            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {6, -20, 9, 100, 101};
            int[] shape2 = {5};
            var expectedRoundTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualRoundTensor = tensor.Round();

            for (int i = 0; i < actualRoundTensor.Size; i++)
            {
                Assert.AreEqual(expectedRoundTensor[i], actualRoundTensor[i]);
            }
        }

        [Test]
        public void Round_()
        {
            float[] data1 = {5.89221f, -20.11f, 9.0f, 100.4999f, 100.5001f};
            int[] shape1 = {5};
            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {6, -20, 9, 100, 101};
            int[] shape2 = {5};
            var expectedRoundTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor.Round(inline: true);

            for (int i = 0; i < expectedRoundTensor.Size; i++)
            {
                Assert.AreEqual(expectedRoundTensor[i], tensor[i]);
            }
        }

        [Test]
        public void Rsqrt()
        {
            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = {4};

            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            var result = tensor1.Rsqrt();

            float[] data2 = {1, (float) 0.7071068, (float) 0.5773503, (float) 0.5};
            int[] shape2 = {4};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            for (int i = 0; i < tensor1.Data.Length; i++)
            {
                Assert.AreEqual(expectedTensor[i], result[i], 1e-3);
            }
        }

        [Test]
        public void Rsqrt_()
        {
            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, (float) 0.7071068, (float) 0.5773503, (float) 0.5};
            int[] shape2 = {4};
            var expectedExpTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Rsqrt(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedExpTensor[i], tensor1[i], 1e-3);
            }
        }

        [Test]
        public void Sigmoid()
        {
            float[] data1 = {0.0f};
            int[] shape1 = {1};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1).Sigmoid();

            float[] data2 = {0.5f};
            int[] shape2 = {1};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);
            Assert.AreEqual(expectedTensor[0], tensor1[0]);

            float[] data3 = {0.1f, 0.5f, 1.0f, 2.0f};
            int[] shape3 = {4};
            float[] data4 = {-0.1f, -0.5f, -1.0f, -2.0f};
            int[] shape4 = {4};
            // Verifies sum of function with inverse x adds up to 1
            var tensor3 = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);
            var tensor4 = ctrl.floatTensorFactory.Create(_data: data4, _shape: shape4);
            var sum = tensor3.Sigmoid().Add(tensor4.Sigmoid());

            float[] data5 = {1.0f, 1.0f, 1.0f, 1.0f};
            int[] shape5 = {4};
            var expectedTensor2 = ctrl.floatTensorFactory.Create(_data: data5, _shape: shape5);

            for (int i = 0; i < sum.Size; i++)
            {
                Assert.AreEqual(sum.Data[i], expectedTensor2.Data[i]);
            }
        }

        [Test]
        public void Sigmoid_()
        {
            float[] data1 = {0.0f};
            int[] shape1 = {1};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            tensor1.Sigmoid(inline: true);

            float[] data2 = {0.5f};
            int[] shape2 = {1};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);
            Assert.AreEqual(expectedTensor[0], tensor1[0]);

            float[] data3 = {0.1f, 0.5f, 1.0f, 2.0f};
            int[] shape3 = {4};
            float[] data4 = {-0.1f, -0.5f, -1.0f, -2.0f};
            int[] shape4 = {4};
            // Verifies sum of function with inverse x adds up to 1
            var tensor3 = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);
            var tensor4 = ctrl.floatTensorFactory.Create(_data: data4, _shape: shape4);
            tensor3.Sigmoid(inline: true);
            tensor4.Sigmoid(inline: true);
            var sum = tensor3.Add(tensor4);

            float[] data5 = {1.0f, 1.0f, 1.0f, 1.0f};
            int[] shape5 = {4};
            var expectedTensor2 = ctrl.floatTensorFactory.Create(_data: data5, _shape: shape5);

            for (int i = 0; i < sum.Size; i++)
            {
                Assert.AreEqual(sum.Data[i], expectedTensor2.Data[i]);
            }
        }

        [Test]
        public void Sign()
        {
            float[] data1 =
                {float.MinValue, -100.0f, -1.0f, -0.0001f, -0.0f, +0.0f, 0.0001f, 1.0f, 10.0f, float.MaxValue};
            int[] shape1 = {1, 10};

            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {-1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape1);

            var result1 = tensor1.Sign();

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], result1[i]);
            }
        }

        [Test]
        public void Sign_()
        {
            float[] data1 =
                {float.MinValue, -100.0f, -1.0f, -0.0001f, -0.0f, +0.0f, 0.0001f, 1.0f, 10.0f, float.MaxValue};
            int[] shape1 = {1, 10};

            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {-1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape1);

            tensor1.Sign(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], tensor1[i]);
            }
        }

        [Test]
        public void Sin()
        {
            float[] data1 = {0.4f, 0.5f, 0.3f, -0.1f};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {0.38941834f, 0.47942554f, 0.29552021f, -0.09983342f};
            int[] shape2 = {4};
            var expectedSinTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualSinTensor = tensor1.Sin();

            for (int i = 0; i < actualSinTensor.Size; i++)
            {
                Assert.AreEqual(expectedSinTensor[i], actualSinTensor[i]);
            }
        }

        [Test]
        public void Sin_()
        {
            float[] data1 = {0.4f, 0.5f, 0.3f, -0.1f};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {0.38941834f, 0.47942554f, 0.29552021f, -0.09983342f};
            int[] shape2 = {4};
            var expectedSinTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Sin(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedSinTensor[i], tensor1[i]);
            }
        }

        [Test]
        public void Sinh()
        {
            float[] data1 = {-0.6366f, 0.2718f, 0.4469f, 1.3122f};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {-0.68048f, 0.27516f, 0.46193f, 1.72255f};
            int[] shape2 = {4};
            var expectedSinhTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualSinhTensor = tensor1.Sinh();

            for (int i = 0; i < actualSinhTensor.Size; i++)
            {
                Assert.AreEqual(expectedSinhTensor[i], actualSinhTensor[i], 1e-5);
            }
        }

        [Test]
        public void Sinh_()
        {
            float[] data1 = {-0.6366f, 0.2718f, 0.4469f, 1.3122f};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {-0.68048f, 0.27516f, 0.46193f, 1.72255f};
            int[] shape2 = {4};
            var expectedSinhTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Sinh(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedSinhTensor[i], tensor1[i], 1e-5);
            }
        }
/* TODO: not sure why this exists... what was SizeTensor?
        [Test]
        public void SizeTensor()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape1 = {2, 3, 2};
            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            var actualSizeTensor = tensor.SizeTensor();

            float[] data2 = {2, 3, 2};
            int[] shape2 = {3};
            var expectedSizeTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            for (int i = 0; i < shape1.Length; i++)
            {
                Assert.AreEqual(actualSizeTensor[i], expectedSizeTensor[i]);
            }
        }*/

        [Test]
        public void Softmax1D()
        {
            float[] data = {1, 2, 3, 4};
            int[] shape = {4};

            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            var actualTensor = tensor.Softmax();

            var expectedTensor = new float[]
                {(float) 0.0320586, (float) 0.08714432, (float) 0.23688282, (float) 0.64391426};
            for (var i = 0; i < expectedTensor.Length; i++)
            {
                Assert.AreEqual(expectedTensor[i], actualTensor[i], 1e-3);
            }
        }
        
        [Test]
        public void Softmax1DAutoGrad()
        {
            float[] data = {(float)1, (float)0.7, (float)0.5, (float)0.3};
            float[] gradData = {1, 0, 0, 0};
            int[] shape = {4};

            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape, _autograd:true);
            var gradTensor = ctrl.floatTensorFactory.Create(_data: gradData, _shape: shape);

            var outputTensor = tensor.Softmax();

            var expectedTensor = new float[]
                {(float) 0.2280, (float) -0.0916, (float) -0.0750, (float) -0.0614};
            
            outputTensor.Backward(gradTensor, null);
            for (var i = 0; i < expectedTensor.Length; i++)
            {
                Assert.AreEqual(expectedTensor[i], tensor.Grad[i], 1e-3);
            }
        }

        [Test]
        public void Softmax2D()
        {
            float[] data = {1, 2, 3, 4};
            int[] shape = {2, 2};

            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            var actualTensor = tensor.Softmax();

            var expectedData = new float[] {(float) 0.2689, (float) 0.7311, (float) 0.2689, (float) 0.7311};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: expectedData, _shape: shape);
            for (var i = 0; i < expectedTensor.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], actualTensor[i], 1e-3);
            }

            actualTensor = tensor.Softmax(0);

            expectedData = new float[] {(float) 0.1192, (float) 0.1192, (float) 0.8808, (float) 0.8808};
            expectedTensor = ctrl.floatTensorFactory.Create(_data: expectedData, _shape: shape);
            for (var i = 0; i < expectedTensor.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], actualTensor[i], 1e-3);
            }
        }

        [Test]
        public void Softmax3D()
        { 
            float[] data = {1, 2, 3, 4, 5, 6, 7, 8};
            int[] shape = {2, 2, 2};

            float[] expectedData =
            {
                (float) 0.2689, (float) 0.7311, (float) 0.2689, (float) 0.7311,
                (float) 0.2689, (float) 0.7311, (float) 0.2689, (float) 0.7311
            };

            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);
            var actualTensor = tensor.Softmax();

            var expectedTensor = ctrl.floatTensorFactory.Create(_data: expectedData, _shape: shape);
            for (var i = 0; i < expectedTensor.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], actualTensor[i], 1e-3);
            }

            actualTensor = tensor.Softmax(1);
            expectedData = new float[]
            {
                (float) 0.1192, (float) 0.1192, (float) 0.8808, (float) 0.8808, (float) 0.1192, (float) 0.1192,
                (float) 0.8808, (float) 0.8808
            };
            expectedTensor = ctrl.floatTensorFactory.Create(_data: expectedData, _shape: shape);
            for (var i = 0; i < expectedTensor.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], actualTensor[i], 1e-3);
            }
            
            actualTensor = tensor.Softmax(0);
            expectedData = new float[]
            {
                (float) 0.0180, (float) 0.0180, (float) 0.0180, (float) 0.0180, (float) 0.9820, (float) 0.9820,
                (float) 0.9820, (float) 0.9820
            };
            expectedTensor = ctrl.floatTensorFactory.Create(_data: expectedData, _shape: shape);
            for (var i = 0; i < expectedTensor.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], actualTensor[i], 1e-3);
            }
        }

        [Test]
        public void Squeeze()
        {
            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = {2, 1, 2, 1};

            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            var newTensor = tensor.Squeeze();

            Assert.AreEqual(2, newTensor.Shape[0]);
            Assert.AreEqual(2, newTensor.Shape[1]);
            Assert.AreEqual(2, newTensor.Shape.Length);

            var anotherTensor = tensor.Squeeze(dim: 3);

            Assert.AreEqual(2, anotherTensor.Shape[0]);
            Assert.AreEqual(1, anotherTensor.Shape[1]);
            Assert.AreEqual(3, anotherTensor.Shape.Length);
        }

        [Test]
        public void Squeeze_()
        {
            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = {2, 1, 2, 1};

            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            tensor1.Squeeze(inline: true);

            Assert.AreEqual(2, tensor1.Shape[0]);
            Assert.AreEqual(2, tensor1.Shape[1]);
            Assert.AreEqual(2, tensor1.Shape.Length);

            float[] data2 = {1, 2, 3, 4};
            int[] shape2 = {2, 1, 2, 1};

            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor2.Squeeze(dim: 3, inline: true);

            Assert.AreEqual(2, tensor2.Shape[0]);
            Assert.AreEqual(1, tensor2.Shape[1]);
            Assert.AreEqual(3, tensor2.Shape.Length);
        }

        [Test]
        public void Sqrt()
        {
            float[] data1 = {float.MaxValue, float.MinValue, 1f, 4f, 5f, 2.3232f, -30f};
            int[] shape1 = {7};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1.8446743E+19f, float.NaN, 1f, 2f, 2.236068f, 1.524205f, float.NaN};
            int[] shape2 = {7};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualTensor = tensor1.Sqrt();

            for (int i = 0; i < expectedTensor.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], actualTensor[i], 1e-3);
            }
        }

        [Test]
        public void Sqrt_()
        {
            float[] data1 = {float.MaxValue, float.MinValue, 1f, 4f, 5f, 2.3232f, -30f};
            int[] shape1 = {7};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1.8446743E+19f, float.NaN, 1f, 2f, 2.236068f, 1.524205f, float.NaN};
            int[] shape2 = {7};
            var expectedExpTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Sqrt(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedExpTensor[i], tensor1[i], 1e-3);
            }
        }


        [Test]
        public void SubtractElementwise()
        {
            float[] data1 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape1 = {2, 4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {float.MaxValue, 10, 1.5f, 0, -1.5f, -10, -20, float.MinValue};
            int[] shape2 = {2, 4};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            float[] data3 = {float.NegativeInfinity, -20, -3, 0, 3, 20, 40, float.PositiveInfinity};
            int[] shape3 = {2, 4};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);

            var result = tensor1.Sub(tensor2);

            for (int i = 0; i < result.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], result[i]);
            }
        }

        [Test]
        public void SubtractElementwise_()
        {
            float[] data1 = {float.MinValue, -10, -1.5f, 0, 1.5f, 10, 20, float.MaxValue};
            int[] shape1 = {2, 4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {float.MaxValue, 10, 1.5f, 0, -1.5f, -10, -20, float.MinValue};
            int[] shape2 = {2, 4};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            float[] data3 = {float.NegativeInfinity, -20, -3, 0, 3, 20, 40, float.PositiveInfinity};
            int[] shape3 = {2, 4};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);

            tensor1.Sub(tensor2, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], tensor1[i]);
            }
        }

        [Test]
        public void SubtractElementwiseUnequalDimensions()
        {
            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4};
            int[] shape2 = {2, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Sub(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void SubtractElementwiseUnequalDimensions_()
        {
            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4};
            int[] shape2 = {2, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Sub(tensor2, inline: true),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void SubtractElementwiseUnequalShapes()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6};
            int[] shape1 = {2, 3};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6};
            int[] shape2 = {3, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Sub(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void SubtractElementwiseUnequalShapes_()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6};
            int[] shape1 = {2, 3};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6};
            int[] shape2 = {3, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Sub(tensor2, inline: true),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void SubtractElementwiseUnequalSizes()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape2 = {2, 6};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Sub(tensor2),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void SubtractElementwiseUnequalSizes_()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape2 = {2, 6};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor1.Sub(tensor2, inline: true),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void SubtractScalar()
        {
            float[] data1 = {-1, 0, 0.1f, 1, float.MaxValue, float.MinValue};
            int[] shape1 = {3, 2};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {-101, -100, -99.9f, -99, float.MaxValue - 100, float.MinValue - 100};
            int[] shape2 = {3, 2};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            float scalar = 100;
            var tensor3 = tensor1.Sub(scalar);

            for (int i = 0; i < tensor3.Size; i++)
            {
                Assert.AreEqual(expectedTensor.Data[i], tensor3.Data[i]);
            }
        }

        [Test]
        public void SubtractScalar_()
        {
            float[] data1 = {-1, 0, 0.1f, 1, float.MaxValue, float.MinValue};
            int[] shape1 = {5, 1};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {-101, -100, -99.9f, -99, float.MaxValue - 100, float.MinValue - 100};
            int[] shape2 = {5, 1};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            float scalar = 100;

            tensor1.Sub(scalar, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor.Data[i], tensor1.Data[i]);
            }
        }

        [Test]
        public void Sum()
        {
            float[] data = new float[]
            {
                1, 2, 3,
                4, 5, 6
            };

            int[] shape = new int[] {2, 3};
            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            Console.WriteLine("tensor {0}", string.Join(", ", tensor.Data));

            Syft.Tensor.FloatTensor result = tensor.Sum();

            Console.WriteLine("tensor.Sum() {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 1);
            Assert.AreEqual(result.Size, 1);
            Assert.AreEqual(result[0], 21.0);

            result = tensor.Sum(0);

            Console.WriteLine("tensor.Sum(0) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 3);
            Assert.AreEqual(result.Size, 3);
            Assert.AreEqual(result[0], 5.0);
            Assert.AreEqual(result[1], 7.0);
            Assert.AreEqual(result[2], 9.0);

            result = tensor.Sum(1);

            Console.WriteLine("tensor.Sum(1) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 1);
            Assert.AreEqual(result.Shape[0], 2);
            Assert.AreEqual(result.Size, 2);
            Assert.AreEqual(result[0], 6.0);
            Assert.AreEqual(result[1], 15.0);

            /*result = tensor.Sum(0, true);

            Console.WriteLine("tensor.Sum(0, true) {0}", string.Join(", ", result.Data));
            Assert.AreEqual(result.Shape.Length, 2);
            Assert.AreEqual(result.Shape[0], 1);
            Assert.AreEqual(result.Shape[1], 3);
            Assert.AreEqual(result.Size, 3);
            Assert.AreEqual(result[0], 5.0);
            Assert.AreEqual(result[1], 7.0);
            Assert.AreEqual(result[2], 9.0);*/
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
        public void Tan_()
        {
            float[] data1 = {30, 20, 40, 50};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {-6.4053312f, 2.23716094f, -1.11721493f, -0.27190061f};
            int[] shape2 = {4};
            var expectedTanTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Tan(inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTanTensor[i], tensor1[i]);
            }
        }

        [Test]
        public void Tanh()
        {
            float[] data1 = {-0.6366f, 0.2718f, 0.4469f, 1.3122f};
            int[] shape1 = {4};
            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {-0.562580109f, 0.265298963f, 0.419347495f, 0.86483103f};
            int[] shape2 = {4};
            var expectedTanhTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualTanhTensor = tensor.Tanh();

            for (int i = 0; i < actualTanhTensor.Size; i++)
            {
                Assert.AreEqual(expectedTanhTensor[i], actualTanhTensor[i]);
            }
        }

        [Test]
        public void TensorId()
        {
            float[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape = {2, 3, 2};

            var tensor1 = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            Assert.AreNotEqual(tensor1.Id, tensor2.Id);
            Assert.AreEqual(tensor1.Id + 1, tensor2.Id);
        }

        [Test]
        public void Trace()
        {
            // test #1
            float[] data1 = {1.2f, 2, 3, 4};
            int[] shape1 = {2, 2};
            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            float actual = tensor.Trace();
            float expected = 5.2f;

            Assert.AreEqual(expected, actual);

            // test #2
            float[] data3 = {1, 2, 3};
            int[] shape3 = {3};
            var non2DTensor = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);
            Assert.That(() => non2DTensor.Trace(),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void Transpose2D()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            int[] shape1 = {2, 5};

            float[] data2 = {1, 6, 2, 7, 3, 8, 4, 9, 5, 10};
            int[] shape2 = {5, 2};

            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            var transpose = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                for (int j = 0; j < tensor.Shape[1]; j++)
                {
                    Assert.AreEqual(tensor[i, j], transpose[j, i]);
                }
            }

            var transposed = tensor.Transpose();
            for (int i = 0; i < tensor.Shape[1]; i++)
            {
                for (int j = 0; j < tensor.Shape[0]; j++)
                {
                    Assert.AreEqual(transposed[i, j], transpose[i, j]);
                }
            }
        }

        [Test]
        public void Transpose3D()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape1 = {3, 2, 2};

            float[] data2 = {1, 5, 9, 3, 7, 11, 2, 6, 10, 4, 8, 12};
            int[] shape2 = {2, 2, 3};

            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            var transpose = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                for (int j = 0; j < tensor.Shape[1]; j++)
                {
                    for (int k = 0; k < tensor.Shape[2]; k++)
                    {
                        Assert.AreEqual(tensor[i, j, k], transpose[k, j, i]);
                    }
                }
            }

            var transposed = tensor.Transpose(0, 2);
            for (int i = 0; i < transposed.Shape[0]; i++)
            {
                for (int j = 0; j < transposed.Shape[1]; j++)
                {
                    for (int k = 0; k < transposed.Shape[2]; k++)
                    {
                        Assert.AreEqual(transposed[i, j, k], transpose[i, j, k]);
                    }
                }
            }
        }

        [Test]
        public void TransposeDimensionsOutOfRange()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape1 = {3, 2, 2};

            // Test negative dimension indexes
            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            Assert.That(() => tensor.Transpose(-1, 0),
                Throws.TypeOf<ArgumentOutOfRangeException>());
            Assert.That(() => tensor.Transpose(0, -1),
                Throws.TypeOf<ArgumentOutOfRangeException>());

            // Test dimension indexes bigger than tensor's shape lenght
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            Assert.That(() => tensor2.Transpose(3, 0),
                Throws.TypeOf<ArgumentOutOfRangeException>());
            Assert.That(() => tensor2.Transpose(0, 3),
                Throws.TypeOf<ArgumentOutOfRangeException>());
        }

        [Test]
        public void TransposeNoDimensionsSpecified()
        {
            float[] data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            int[] shape1 = {3, 2, 2};

            // Test Tensor with more than 2 dimensions
            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            Assert.That(() => tensor.Transpose(),
                Throws.TypeOf<InvalidOperationException>());

            // Test tensor with less than 2 dimensions
            float[] data2 = {1, 2, 3, 4, 5};
            int[] shape2 = {5};
            tensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);
            Assert.That(() => tensor.Transpose(),
                Throws.TypeOf<InvalidOperationException>());
        }

        public void Triu_()
        {
            int k = 0;

            // Test tensor with dimension < 2
            float[] data1 = {1, 2, 3, 4, 5, 6};
            int[] shape1 = {6};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);
            Assert.That(() => tensor1.Triu_(k),
                Throws.TypeOf<InvalidOperationException>());

            // Test tensor with dimension > 2
            float[] data2 = {1, 2, 3, 4, 5, 6, 7, 8};
            int[] shape2 = {2, 2, 2};
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            Assert.That(() => tensor2.Triu_(k),
                Throws.TypeOf<InvalidOperationException>());

            // Test dim = 2, k = 0
            k = 0;
            float[] data3 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
            int[] shape3 = {3, 3};
            var tensor3 = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);
            tensor3.Triu_(k);
            float[] data3Triu = {1, 2, 3, 0, 5, 6, 0, 0, 9};
            var tensor3Triu = ctrl.floatTensorFactory.Create(_data: data3Triu, _shape: shape3);
            for (int i = 0; i < tensor3.Size; i++)
            {
                Assert.AreEqual(tensor3[i], tensor3Triu[i]);
            }

            // Test dim = 2, k = 2
            k = 2;
            float[] data4 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
            int[] shape4 = {3, 3};
            var tensor4 = ctrl.floatTensorFactory.Create(_data: data4, _shape: shape4);
            tensor4.Triu_(k);
            float[] data4Triu = {0, 0, 3, 0, 0, 0, 0, 0, 0};
            var tensor4Triu = ctrl.floatTensorFactory.Create(_data: data4Triu, _shape: shape4);
            for (int i = 0; i < tensor4.Size; i++)
            {
                Assert.AreEqual(tensor4[i], tensor4Triu[i]);
            }

            // Test dim = 2, k = -1
            k = -1;
            float[] data5 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
            int[] shape5 = {3, 3};
            var tensor5 = ctrl.floatTensorFactory.Create(_data: data5, _shape: shape5);
            tensor5.Triu_(k);
            float[] data5Triu = {1, 2, 3, 4, 5, 6, 0, 8, 9};
            var tensor5Triu = ctrl.floatTensorFactory.Create(_data: data5Triu, _shape: shape5);
            for (int i = 0; i < tensor5.Size; i++)
            {
                Assert.AreEqual(tensor5[i], tensor5Triu[i]);
            }

            // Test dim = 2, k >> ndims
            k = 100;
            float[] data6 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
            int[] shape6 = {3, 3};
            var tensor6 = ctrl.floatTensorFactory.Create(_data: data6, _shape: shape6);
            tensor6.Triu_(k);
            float[] data6Triu = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            var tensor6Triu = ctrl.floatTensorFactory.Create(_data: data6Triu, _shape: shape6);
            for (int i = 0; i < tensor6.Size; i++)
            {
                Assert.AreEqual(tensor6[i], tensor6Triu[i]);
            }

            // Test dim = 2, k << ndims
            k = -100;
            float[] data7 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
            int[] shape7 = {3, 3};
            var tensor7 = ctrl.floatTensorFactory.Create(_data: data7, _shape: shape7);
            tensor7.Triu_(k);
            float[] data7Triu = {1, 2, 3, 4, 5, 6, 7, 8, 9};
            var tensor7Triu = ctrl.floatTensorFactory.Create(_data: data7Triu, _shape: shape7);
            for (int i = 0; i < tensor7.Size; i++)
            {
                Assert.AreEqual(tensor7[i], tensor7Triu[i]);
            }
        }

        [Test]
        public void Trunc()
        {
            float[] data = {-0.323232f, 0.323893f, 0.99999f, 1.2323389f};
            int[] shape = {4};
            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            float[] truncatedData = {-0f, 0f, 0f, 1f};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: truncatedData, _shape: shape);

            var truncatedTensor = tensor.Trunc();
            for (int i = 0; i < truncatedTensor.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], truncatedTensor[i]);
            }
        }

        [Test]
        public void Unsqueeze()
        {
            float[] data1 = {1, 2, 3, 4};
            int[] shape1 = {2, 2};

            var tensor = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            var newTensor = tensor.Unsqueeze(1);

            Assert.AreEqual(1, newTensor.Shape[1]);
            Assert.AreEqual(3, newTensor.Shape.Length);
        }

        [Test]
        public void View()
        {
            float[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            int[] shape = {4, 4};

            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            int[] newShape = {2, 8};

            var newTensor = tensor.View(newShape);

            Assert.AreEqual(2, newTensor.Shape[0]);
            Assert.AreEqual(8, newTensor.Shape[1]);

            Assert.AreEqual(8, newTensor.Strides[0]);
            Assert.AreEqual(1, newTensor.Strides[1]);
        }

        [Test]
        public void View_()
        {
            float[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            int[] shape = {4, 4};

            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            int[] newShape = {2, 8};

            tensor.View(new_shape:newShape, inline: true);

            Assert.AreEqual(2, tensor.Shape[0]);
            Assert.AreEqual(8, tensor.Shape[1]);

            Assert.AreEqual(8, tensor.Strides[0]);
            Assert.AreEqual(1, tensor.Strides[1]);
        }

        [Test]
        public void Stride()
        {
            float[] data = {1, 2, 3, 4, 5, 6};
            int[] shape = {1, 2, 3};

            var tensor = ctrl.floatTensorFactory.Create(_data: data, _shape: shape);

            var strides = tensor.Strides;

            Assert.AreEqual(6, strides[0]);
            Assert.AreEqual(3, strides[1]);
            Assert.AreEqual(1, strides[2]);
        }

        [Test]
        public void Zero_()
        {
            float[] data1 = {-1, 0, 1, float.MaxValue, float.MinValue};
            int[] shape1 = {5};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {0, 0, 0, 0, 0};
            int[] shape2 = {5};
            var expectedTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Zero_();

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], tensor1[i]);
            }
        }


/* closes class and namespace */
    }
}