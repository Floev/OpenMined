using NUnit.Framework;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Layer;
using UnityEngine;

namespace OpenMined.Test.Model
{
    [Category("LayerCPUTests")]
    public class Layer
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
        public void Conv2d()
        {
            float[] inputData = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            int[] inputShape = new int[] { 1, 1, 3, 3 };
            int[] kernel = new int[] { 2, 2 };
            float[] kernelData = new float[] { 1, 1, 1, 1 };
            float[] outputData = new float[] { 12, 16, 24, 28 };
            int[] outputShape = new int[] { 1, 1, 2, 2 };
//            float[] biasData = new float[] { -1 };
//            float[] biasedOutputData = new float[] { 11, 15, 23, 27 };

            var input = ctrl.floatTensorFactory.Create(_data: inputData, _shape: inputShape);
            var target = ctrl.floatTensorFactory.Create(_data: outputData, _shape: outputShape);

            var conv = new Conv2d(ctrl, 2, 2, kernel, kernelData: kernelData);
//            var biasedConv = new Conv2d(ctrl, 2, 2, kernel, kernelData: kernelData,biasData:biasData);
            var output = conv.Forward(input);
//            var biasedOutput = biasedConv.Forward(input);
            for (int i = 0; i < target.Size; i++)
            {
                Assert.AreEqual(output[i], target[i]);
//                Assert.AreEqual(biasedOutput[i], biasedOutputData[i]);
            }

            float[] inputData2 = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
            int[] inputShape2 = new int[] { 1, 2, 3, 3 };
            var input2 = ctrl.floatTensorFactory.Create(_data: inputData2, _shape: inputShape2);
            float[] outputData2 = new float[] { 12, 16, 24, 28, 48, 52, 60, 64 };
            int[] outputShape2 = new int[] { 1, 2, 2, 2 };
//            float[] biasData2 = new float[] { -1, 7 };
//            float[] biasedOutputData2 = new float[] { 11, 15, 23, 27, 55, 59, 67, 71 };

            var target2 = ctrl.floatTensorFactory.Create(_data: outputData2, _shape: outputShape2);

            var conv2 = new Conv2d(ctrl, 2, 2, kernel, kernelData: kernelData);
//            var biasedConv2 = new Conv2d(ctrl, 2, 2, kernel, kernelData: kernelData,biasData:biasData2);
            var output2 = conv2.Forward(input2);
//            var biasedOutput2 = biasedConv2.Forward(input2);
            for (int i = 0; i < target2.Size; i++)
            {
                Assert.AreEqual(output2[i], target2[i]);
//                Assert.AreEqual(biasedOutput2[i], biasedOutputData2[i]);
            }

            float[] inputData3 = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            int[] inputShape3 = new int[] { 1, 1, 3, 3 };
            int[] kernel3 = new int[] { 3, 2 };
            float[] kernelData3 = new float[] { 1, 1, 1, 1, 1, 1 };
            var input3 = ctrl.floatTensorFactory.Create(_data: inputData3, _shape: inputShape3);
            float[] outputData3 = new float[] { 27, 33 };
            int[] outputShape3 = new int[] { 1, 1, 1, 2 };

            var target3 = ctrl.floatTensorFactory.Create(_data: outputData3, _shape: outputShape3);

            var conv3 = new Conv2d(ctrl, 2, 2, kernel3, kernelData: kernelData3);
            var output3 = conv3.Forward(input3);
            for (int i = 0; i < output3.Size; i++)
            {
                Assert.AreEqual(output3[i], target3[i]);
            }

            //now backprop (with random weights!!)
            var x = ctrl.floatTensorFactory.Create(_data: new float[] { 1, 0, 0, 1 }, _shape: new int[] { 1, 1, 2, 2 }, _autograd:true);
            var y = ctrl.floatTensorFactory.Create(_data: new float[] { 2, 1 }, _shape: new int[] { 1, 1, 1, 2 }, _autograd: true);
            var conv4 = new Conv2d(ctrl, 1, 1, new int[] { 2, 1 });
            var prediction = conv4.Forward(x);
            var err = prediction.Sub(y);
            err.Autograd = false;
            prediction.Backward(err);
            var param = ctrl.floatTensorFactory.Get(conv4.getParameter(0));
            param.Sub(param.Grad, inline: true);
            var correct_prediction = conv4.Forward(x);
            for (int i = 0; i < correct_prediction.Size; i++)
            {
                Assert.AreEqual(correct_prediction[i], y[i],1e-7);
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

            var input = ctrl.floatTensorFactory.Create(_data: inputData, _shape: inputShape);
            var target = ctrl.floatTensorFactory.Create(_data: outputData, _shape: outputShape);
            var conv = new Conv2d(ctrl, 2, 2, kernel, transposed: true, kernelData: kernelData);
            var output = conv.Forward(input);
            for (int i = 0; i < output.Size; i++)
            {
                Assert.AreEqual(output[i], target[i]);
            }

            int[] kernel2 = new int[] { 3, 1 };
            float[] kernelData2 = new float[] { 1, 1, 1 };
            float[] outputData2 = new float[] { 1, 2, 4, 6, 4, 6, 3, 4 };
            int[] outputShape2 = new int[] { 1, 1, 4, 2 };

            var target2 = ctrl.floatTensorFactory.Create(_data: outputData2, _shape: outputShape2);
            var conv2 = new Conv2d(ctrl, 2, 2, kernel2, transposed: true, kernelData: kernelData2);
            var output2 = conv2.Forward(input);
            for (int i = 0; i < output2.Size; i++)
            {
                Assert.AreEqual(output2[i], target2[i]);
            }
        }

        [Test]
        public void Linear()
        {
            float[] inputData = new float[] { 1, 2, 3, 4, 5, 6 };
            int[] inputShape = new int[] { 2, 3 };//2 samples of 3
            float[] weightsData = new float[] { 4, 5, 6, 7, 8, 9 };//3x2
            float[] biasData = new float[] { 1, -1 };
            float[] outputData = new float[] { 40, 46, 94, 109 };
            int[] outputShape = new int[] { 2, 2 };
            float[] biasedOutputData = new float[] { 41, 45, 95, 108 };
            
            var input = ctrl.floatTensorFactory.Create(_data: inputData, _shape: inputShape,_autograd:true);
            var target = ctrl.floatTensorFactory.Create(_data: outputData, _shape: outputShape);
            var biasedTarget = ctrl.floatTensorFactory.Create(_data: biasedOutputData, _shape: outputShape);

            var linear = new Linear(ctrl, 3, 2, weights: weightsData);
            var biasedLinear = new Linear(ctrl, 3, 2, weights: weightsData,bias:biasData);

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
            var linear2 = new Linear(ctrl, 2, 2);
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
                Assert.AreEqual(correct_prediction1[i], y1[i],1e-6);
                Assert.AreEqual(correct_prediction2[i], y2[i],1e-6);
            }
        }
    }
}