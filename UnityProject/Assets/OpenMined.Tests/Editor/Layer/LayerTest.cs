using NUnit.Framework;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Layer;

namespace OpenMined.Tests.Model
{
    [Category("LayerCPUTests")]
    public class LayerTest
    {
        public SyftController ctrl;

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
        public void Linear()
        {
            bool[] fast_speeds = { false, true };
            foreach (bool fast_speed in fast_speeds)
            {
                float[] inputData = new float[] { 1, 2, 3, 4, 5, 6 };
                int[] inputShape = new int[] { 2, 3 };//2 samples of 3
                float[] weightsData = new float[] { 4, 5, 6, 7, 8, 9 };//3x2
                float[] biasData = new float[] { 1, -1 };
                float[] outputData = new float[] { 40, 46, 94, 109 };
                int[] outputShape = new int[] { 2, 2 };
                float[] biasedOutputData = new float[] { 41, 45, 95, 108 };

                var input = ctrl.floatTensorFactory.Create(_data: inputData, _shape: inputShape, _autograd: true);
                var target = ctrl.floatTensorFactory.Create(_data: outputData, _shape: outputShape);
                var biasedTarget = ctrl.floatTensorFactory.Create(_data: biasedOutputData, _shape: outputShape);

                var linear = new Linear(ctrl, 3, 2, weights: weightsData, fast: fast_speed);
                var biasedLinear = new Linear(ctrl, 3, 2, weights: weightsData, bias: biasData, fast: fast_speed);

                Assert.True(linear.getParameterCount() == 6);
                Assert.True(biasedLinear.getParameterCount() == 8);

                var output = linear.Forward(input);
                var biasedOutput = biasedLinear.Forward(input);

                for (int i = 0; i < target.Size; i++)
                {
                    Assert.AreEqual(output[i], target[i]);
                    Assert.AreEqual(biasedOutput[i], biasedTarget[i]);
                }

                //running in batch mode, so testing "two gradients!
                var grad = ctrl.floatTensorFactory.Create(_data: new float[] { 1, 1, 0, -2 }, _shape: new int[] { 2, 2 });
                var grad2 = ctrl.floatTensorFactory.Create(_data: new float[] { 1, 1, 0, -2 }, _shape: new int[] { 2, 2 });
                output.Backward(grad);

                biasedOutput.Backward(grad2);

                float[] weightGradTarget = new float[] { 1, -7, 2, -8, 3, -9 };
                float[] biasGradTarget = new float[] { 1, -1 };

                var weightGrad = ctrl.floatTensorFactory.Get(linear.getParameter(0)).Grad;
                var weightGrad2 = ctrl.floatTensorFactory.Get(biasedLinear.getParameter(0)).Grad;
                var biasGrad = ctrl.floatTensorFactory.Get(biasedLinear.getParameter(1)).Grad;
                if (fast_speed)
                {
                    weightGrad = weightGrad.Transpose();
                    weightGrad2 = weightGrad2.Transpose();
                }

                for (int i = 0; i < weightGrad.Size; i++)
                {
                    Assert.AreEqual(weightGradTarget[i], weightGrad[i]);
                    Assert.AreEqual(weightGrad[i], weightGrad2[i]);
                }

                for (int i = 0; i < biasGrad.Size; i++)
                {
                    Assert.AreEqual(biasGradTarget[i], biasGrad[i]);
                }

                //now backprop (with random weight initiallization!!)
                var x1 = ctrl.floatTensorFactory.Create(_data: new float[] { 1, 0 }, _shape: new int[] { 1, 2 }, _autograd: true);
                var x2 = ctrl.floatTensorFactory.Create(_data: new float[] { 0, 1 }, _shape: new int[] { 1, 2 }, _autograd: true);
                var y1 = ctrl.floatTensorFactory.Create(_data: new float[] { 5, 6 }, _shape: new int[] { 1, 2 }, _autograd: true);
                var y2 = ctrl.floatTensorFactory.Create(_data: new float[] { .3f, -8 }, _shape: new int[] { 1, 2 }, _autograd: true);
                var linear2 = new Linear(ctrl, 2, 2, fast: fast_speed);
                var prediction1 = linear2.Forward(x1);
                var prediction2 = linear2.Forward(x2);
                var err1 = prediction1.Sub(y1);
                err1.Autograd = false;
                var err2 = prediction2.Sub(y2);
                err2.Autograd = false;
                prediction1.Backward(err1);
                prediction2.Backward(err2);
                foreach (int p in linear2.getParameters())
                {
                    var param = ctrl.floatTensorFactory.Get(p);
                    if (param.Grad != null)
                    {
                        param.Sub(param.Grad, inline: true);
                    }
                }

                var correct_prediction1 = linear2.Forward(x1);
                var correct_prediction2 = linear2.Forward(x2);
                for (int i = 0; i < correct_prediction1.Size; i++)
                {
                    Assert.AreEqual(correct_prediction1[i], y1[i], 1e-6);
                    Assert.AreEqual(correct_prediction2[i], y2[i], 1e-6);
                }
            }
        }
    }
}