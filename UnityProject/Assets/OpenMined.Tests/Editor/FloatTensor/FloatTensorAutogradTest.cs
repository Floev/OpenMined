using System;
using NUnit.Framework;
using OpenMined.Network.Controllers;
using OpenMined.Syft.NN;
using UnityEngine;

namespace OpenMined.Tests.Editor.FloatTensor
{
    [Category("FloatTensorAutogradTests")]
    public class FloatTensorAutogradTest
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
        public void AddElemAutograd()
        {
            
            var a = ctrl.floatTensorFactory.Create(_data: new float[]{1,2,3,4,5}, _shape: new int[]{5});
            var b = ctrl.floatTensorFactory.Create(_data: new float[]{5,1,3,9,2}, _shape: new int[]{5});

            a.Autograd = true;
            b.Autograd = true;
            
            var c_expected = ctrl.floatTensorFactory.Create(_data: new float[]{6,3,6,13,7}, _shape: new int[]{5});
            var c_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1}, _shape: new int[]{5});
            
            var ab_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1}, _shape: new int[]{5});

            var c = a.Add(b);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(ab_grad[i], a.Grad[i]);
                Assert.AreEqual(ab_grad[i], b.Grad[i]);
            }
            
            
            // check that repeating doesn't break it
            c = a.Add(b);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(ab_grad[i], a.Grad[i]);
                Assert.AreEqual(ab_grad[i], b.Grad[i]);
            }

            // see if it's allocating new tensors during the forward pass
            ctrl.allow_new_tensors = false;
            
            // check that repeating the forward pass doesn't break it
            c = a.Add(b);
            c = a.Add(b);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(ab_grad[i], a.Grad[i]);
                Assert.AreEqual(ab_grad[i], b.Grad[i]);
            }
            
            // cleanup
            ctrl.allow_new_tensors = true;
           
        }
        
        [Test]
        public void AddScalarAutograd()
        {
            
            var a = ctrl.floatTensorFactory.Create(_data: new float[]{1,2,3,4,5}, _shape: new int[]{5});

            a.Autograd = true;
            
            var c_expected = ctrl.floatTensorFactory.Create(_data: new float[]{6,7,8,9,10}, _shape: new int[]{5});
            var c_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1}, _shape: new int[]{5});
            
            var a_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1}, _shape: new int[]{5});

            var c = a.Add(5);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }
            
            
            // check that repeating doesn't break it
            c = a.Add(5);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }

            // see if it's allocating new tensors during the forward pass
            ctrl.allow_new_tensors = false;
            
            // check that repeating the forward pass doesn't break it
            c = a.Add(5);
            c = a.Add(5);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }
            
            // cleanup
            ctrl.allow_new_tensors = true;
           
        }
        
        [Test]
        public void SubElemAutograd()
        {
            
            var a = ctrl.floatTensorFactory.Create(_data: new float[]{1,2,3,4,5}, _shape: new int[]{5});
            var b = ctrl.floatTensorFactory.Create(_data: new float[]{5,1,3,9,2}, _shape: new int[]{5});

            a.Autograd = true;
            b.Autograd = true;
            
            var c_expected = ctrl.floatTensorFactory.Create(_data: new float[]{-4,1,0,-5,3}, _shape: new int[]{5});
            var c_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1}, _shape: new int[]{5});
            
            var a_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1}, _shape: new int[]{5});
            var b_grad = ctrl.floatTensorFactory.Create(_data: new float[]{-1,-1,-1,-1,-1}, _shape: new int[]{5});

            var c = a.Sub(b);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // subtraction is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
                Assert.AreEqual(b_grad[i], b.Grad[i]);
            }
            
            
            // check that repeating doesn't break it
            c = a.Sub(b);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
                Assert.AreEqual(b_grad[i], b.Grad[i]);
            }

            // see if it's allocating new tensors during the forward pass
            ctrl.allow_new_tensors = false;
            
            // check that repeating the forward pass doesn't break it
            c = a.Sub(b);
            c = a.Sub(b);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
                Assert.AreEqual(b_grad[i], b.Grad[i]);
            }
            
            // cleanup
            ctrl.allow_new_tensors = true;
              
        }
        
        [Test]
        public void SubScalarAutograd()
        {
            
            var a = ctrl.floatTensorFactory.Create(_data: new float[]{1,2,3,4,5}, _shape: new int[]{5});

            a.Autograd = true;
            
            var c_expected = ctrl.floatTensorFactory.Create(_data: new float[]{-4,-3,-2,-1,0}, _shape: new int[]{5});
            var c_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1}, _shape: new int[]{5});
            
            var a_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1}, _shape: new int[]{5});

            var c = a.Sub(5);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // subtraction is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }
            
            
            // check that repeating doesn't break it
            c = a.Sub(5);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }

            // see if it's allocating new tensors during the forward pass
            ctrl.allow_new_tensors = false;
            
            // check that repeating the forward pass doesn't break it
            c = a.Sub(5);
            c = a.Sub(5);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }
            
            // cleanup
            ctrl.allow_new_tensors = true;
           
        }
        
        [Test]
        public void DivElemAutograd()
        {
            
            var a = ctrl.floatTensorFactory.Create(_data: new float[]{1,2,3,4,5}, _shape: new int[]{5});
            var b = ctrl.floatTensorFactory.Create(_data: new float[]{5,1,3,8,2}, _shape: new int[]{5});

            a.Autograd = true;
            b.Autograd = true;
            
            var c_expected = ctrl.floatTensorFactory.Create(_data: new float[]{0.2f,2,1,0.5f,2.5f},_shape: new int[]{5});
            var c_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1}, _shape: new int[]{5});
            
            var a_grad = ctrl.floatTensorFactory.Create(_data: new float[]{0.2f,1,0.3333f,0.125f,0.5f}, _shape: new int[]{5});
            var b_grad = ctrl.floatTensorFactory.Create(_data: new float[]{-0.04f,-2,-0.3333f,-0.0625f,-1.25f}, _shape: new int[]{5});

            var c = a.Div(b);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // division is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.True(Math.Abs(a_grad[i] - a.Grad[i]) < 0.0001);
                Assert.True(Math.Abs(b_grad[i] - b.Grad[i]) < 0.0001);
            }
            
            
            // check that repeating doesn't break it
            c = a.Div(b);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.True(Math.Abs(a_grad[i] - a.Grad[i]) < 0.0001);
                Assert.True(Math.Abs(b_grad[i] - b.Grad[i]) < 0.0001);
            }

            // see if it's allocating new tensors during the forward pass
            ctrl.allow_new_tensors = false;
            
            // check that repeating the forward pass doesn't break it
            c = a.Div(b);
            c = a.Div(b);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.True(Math.Abs(a_grad[i] - a.Grad[i]) < 0.0001);
                Assert.True(Math.Abs(b_grad[i] - b.Grad[i]) < 0.0001);
            }
            
            // cleanup
            ctrl.allow_new_tensors = true;
              
        }
        
        [Test]
        public void DivScalarAutograd()
        {
            
            var a = ctrl.floatTensorFactory.Create(_data: new float[]{1,2,3,4,5}, _shape: new int[]{5});

            a.Autograd = true;
            
            var c_expected = ctrl.floatTensorFactory.Create(_data: new float[]{0.5f,1,1.5f,2,2.5f}, _shape: new int[]{5});
            var c_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1}, _shape: new int[]{5});
            
            var a_grad = ctrl.floatTensorFactory.Create(_data: new float[]{.5f,.5f,.5f,.5f,.5f}, _shape: new int[]{5});

            var c = a.Div(2);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // division is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }
            
            
            // check that repeating doesn't break it
            c = a.Div(2);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }

            // see if it's allocating new tensors during the forward pass
            ctrl.allow_new_tensors = false;
            
            // check that repeating the forward pass doesn't break it
            c = a.Div(2);
            c = a.Div(2);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }
            
            // cleanup
            ctrl.allow_new_tensors = true;
           
        }
        
        [Test]
        public void MulElemAutograd()
        {
            
            var a = ctrl.floatTensorFactory.Create(_data: new float[]{1,2,3,4,5}, _shape: new int[]{5});
            var b = ctrl.floatTensorFactory.Create(_data: new float[]{5,1,3,8,2}, _shape: new int[]{5});

            a.Autograd = true;
            b.Autograd = true;
            
            var c_expected = ctrl.floatTensorFactory.Create(_data: new float[]{5,2,9,32,10},_shape: new int[]{5});
            var c_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1}, _shape: new int[]{5});
            
            var a_grad = ctrl.floatTensorFactory.Create(_data: new float[]{5,1,3,8,2}, _shape: new int[]{5});
            var b_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,2,3,4,5}, _shape: new int[]{5});

            var c = a.Mul(b);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // multiplication is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.True(Math.Abs(a_grad[i] - a.Grad[i]) < 0.0001);
                Assert.True(Math.Abs(b_grad[i] - b.Grad[i]) < 0.0001);
            }
            
            
            // check that repeating doesn't break it
            c = a.Mul(b);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.True(Math.Abs(a_grad[i] - a.Grad[i]) < 0.0001);
                Assert.True(Math.Abs(b_grad[i] - b.Grad[i]) < 0.0001);
            }

            // see if it's allocating new tensors during the forward pass
            ctrl.allow_new_tensors = false;
            
            // check that repeating the forward pass doesn't break it
            c = a.Mul(b);
            c = a.Mul(b);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.True(Math.Abs(a_grad[i] - a.Grad[i]) < 0.0001);
                Assert.True(Math.Abs(b_grad[i] - b.Grad[i]) < 0.0001);
            }
            
            // cleanup
            ctrl.allow_new_tensors = true;
              
        }
        
        [Test]
        public void MulScalarAutograd()
        {
            
            var a = ctrl.floatTensorFactory.Create(_data: new float[]{1,2,3,4,5}, _shape: new int[]{5});

            a.Autograd = true;
            
            var c_expected = ctrl.floatTensorFactory.Create(_data: new float[]{5,10,15,20,25}, _shape: new int[]{5});
            var c_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1}, _shape: new int[]{5});
            
            var a_grad = ctrl.floatTensorFactory.Create(_data: new float[]{5,5,5,5,5}, _shape: new int[]{5});

            var c = a.Mul(5);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // multiplication is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }
            
            
            // check that repeating doesn't break it
            c = a.Mul(5);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }

            // see if it's allocating new tensors during the forward pass
            ctrl.allow_new_tensors = false;
            
            // check that repeating the forward pass doesn't break it
            c = a.Mul(5);
            c = a.Mul(5);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }
            
            // cleanup
            ctrl.allow_new_tensors = true;
           
        }
        
        [Test]
        public void MMAutograd()
        {

            int[] ash = new int[] {2, 5};
            float[] a_data = new float[] {1, 2, 3, 4, 5, 2, 3, 4, 5, 6};
            var a = ctrl.floatTensorFactory.Create(_data: a_data, _shape: ash);

            int[] bs = new int[] {5,3};
            float[] b_data = new float[] {5, 2, 3, 1, 5, 5, 3, 3, 2, 8, 2, 3, 2, 5, 6};
            var b = ctrl.floatTensorFactory.Create(_data: b_data, _shape: bs);

            a.Autograd = true;
            b.Autograd = true;

            int[] ces = new int[] {2,3};
            var c_expected = ctrl.floatTensorFactory.Create(_data: new float[]{58,54,61,77,71,80},_shape: ces);
            var c_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1,1}, _shape: new int[]{2,3});

            float[] a_grad_data = new float[] {10, 11, 8, 13, 13, 10, 11, 8, 13, 13};
            var a_grad = ctrl.floatTensorFactory.Create(_data: a_grad_data, _shape: ash);

            float[] b_grad_data = new float[] {3, 3, 3, 5, 5, 5, 7, 7, 7, 9, 9, 9, 11, 11, 11};
            var b_grad = ctrl.floatTensorFactory.Create(_data: b_grad_data, _shape: bs);

            var c = a.MM(b);
            c.Backward(c_grad);

            for (int i = 0; i < c.Size; i++)
            {
                // multiplication is correct
                Assert.AreEqual(c_expected.Data[i],c.Data[i]);
            }

            for (int i = 0; i < a_grad.Size; i++)
            {
                // a gradients are correct
                Assert.True(Math.Abs(a_grad.Data[i] - a.Grad.Data[i]) < 0.0001);    
            }
            
            for (int i = 0; i < b_grad.Size; i++)
            {
                // a gradients are correct
                Assert.True(Math.Abs(b_grad.Data[i] - b.Grad.Data[i]) < 0.0001);    
            }
            
            // check that repeating doesn't break it
            c = a.MM(b);
            c.Backward(c_grad);

            for (int i = 0; i < c.Size; i++)
            {
                // multiplication is correct
                Assert.AreEqual(c_expected.Data[i],c.Data[i]);
            }

            for (int i = 0; i < a_grad.Size; i++)
            {
                // a gradients are correct
                Assert.True(Math.Abs(a_grad.Data[i] - a.Grad.Data[i]) < 0.0001);    
            }
            
            for (int i = 0; i < b_grad.Size; i++)
            {
                // a gradients are correct
                Assert.True(Math.Abs(b_grad.Data[i] - b.Grad.Data[i]) < 0.0001);    
            }

            // see if it's allocating new tensors during the forward pass
            ctrl.allow_new_tensors = false;
            
            // check that repeating the forward pass doesn't break it
            c = a.MM(b);
            c = a.MM(b);
            c.Backward(c_grad);

            for (int i = 0; i < c.Size; i++)
            {
                // multiplication is correct
                Assert.AreEqual(c_expected.Data[i],c.Data[i]);
            }

            for (int i = 0; i < a_grad.Size; i++)
            {
                // a gradients are correct
                Assert.True(Math.Abs(a_grad.Data[i] - a.Grad.Data[i]) < 0.0001);    
            }
            
            for (int i = 0; i < b_grad.Size; i++)
            {
                // a gradients are correct
                Assert.True(Math.Abs(b_grad.Data[i] - b.Grad.Data[i]) < 0.0001);    
            }
            
            // cleanup
            ctrl.allow_new_tensors = true;
              
        }
        
        [Test]
        public void PowScalarAutograd()
        {
            
            var a = ctrl.floatTensorFactory.Create(_data: new float[]{1,2,3,4,5}, _shape: new int[]{5});

            a.Autograd = true;
            
            var c_expected = ctrl.floatTensorFactory.Create(_data: new float[]{1,4,9,16,25}, _shape: new int[]{5});
            var c_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1}, _shape: new int[]{5});
            
            var a_grad = ctrl.floatTensorFactory.Create(_data: new float[]{2,4,6,8,10}, _shape: new int[]{5});

            var c = a.Pow(2);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // multiplication is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }
            
            
            // check that repeating doesn't break it
            c = a.Pow(2);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }

            // see if it's allocating new tensors during the forward pass
            ctrl.allow_new_tensors = false;
            
            // check that repeating the forward pass doesn't break it
            c = a.Pow(2);
            c = a.Pow(2);
            c.Backward(c_grad);

            for (int i = 0; i < a.Size; i++)
            {
                // sum is correct
                Assert.AreEqual(c_expected[i],c[i]);
                
                // gradients are correct
                Assert.AreEqual(a_grad[i], a.Grad[i]);
            }
            
            // cleanup
            ctrl.allow_new_tensors = true;
           
        }
        
        
        [Test]
        public void TransposeAutograd()
        {

            int[] ash = new int[] {2, 3};
            float[] a_data = new float[] {1, 2, 3, 2, 3, 4};
            var a = ctrl.floatTensorFactory.Create(_data: a_data, _shape: ash);

            a.Autograd = true;

            int[] ces = new int[] {3,2};
            var c_expected = ctrl.floatTensorFactory.Create(_data: new float[]{1,2,2,3,3,4},_shape: ces);
            var c_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,2,2,4,3,6}, _shape: ces);

            float[] a_grad_data = new float[] {1,2,3,2,4,6};
            var a_grad = ctrl.floatTensorFactory.Create(_data: a_grad_data, _shape: ash);

            var c = a.Transpose();
            c.Backward(c_grad);

            for (int i = 0; i < c.Size; i++)
            {
                // multiplication is correct
                Assert.AreEqual(c_expected.Data[i],c.Data[i]);
            }

            for (int i = 0; i < a_grad.Size; i++)
            {
                // a gradients are correct
                Assert.True(Math.Abs(a_grad.Data[i] - a.Grad.Data[i]) < 0.0001);    
            }
            
            
            // check that repeating doesn't break it
            c = a.Transpose();
            c.Backward(c_grad);

            for (int i = 0; i < c.Size; i++)
            {
                // multiplication is correct
                Assert.AreEqual(c_expected.Data[i],c.Data[i]);
            }

            for (int i = 0; i < a_grad.Size; i++)
            {
                // a gradients are correct
                Assert.True(Math.Abs(a_grad.Data[i] - a.Grad.Data[i]) < 0.0001);    
            }

            // see if it's allocating new tensors during the forward pass
            ctrl.allow_new_tensors = false;
            
            // check that repeating the forward pass doesn't break it
            c = a.Transpose();
            c = a.Transpose();
            c.Backward(c_grad);

            for (int i = 0; i < c.Size; i++)
            {
                // multiplication is correct
                Assert.AreEqual(c_expected.Data[i],c.Data[i]);
            }

            for (int i = 0; i < a_grad.Size; i++)
            {
                // a gradients are correct
                Assert.True(Math.Abs(a_grad.Data[i] - a.Grad.Data[i]) < 0.0001);    
            }
            // cleanup
            ctrl.allow_new_tensors = true;
              
        }

        [Test]
        public void SigmoidAutograd()
        {

            int[] ash = new int[] {2, 5};
            float[] a_data = new float[] {1, 2, 3, 4, 5, 2, 3, 4, 5, 6};
            var a = ctrl.floatTensorFactory.Create(_data: a_data, _shape: ash);

            a.Autograd = true;

            int[] ces = ash;
            float[] c_data = new float[] {0.7310586f ,  0.88079703f,  0.95257413f,  0.98201376f,  0.99330717f,
                0.88079703f,  0.95257413f,  0.98201376f,  0.99330717f,  0.99752742f};

            var c_expected = ctrl.floatTensorFactory.Create(_data: c_data,_shape: ces);
            var c_grad = ctrl.floatTensorFactory.Create(_data: new float[]{1,1,1,1,1,1,1,1,1,1}, _shape: ces);

            float[] a_grad_data = new float[] {0.1966f,  0.1050f,  0.0452f,  0.0177f,  0.0066f,
                0.1050f,  0.0452f,  0.0177f,  0.0066f,  0.0025f};
            var a_grad = ctrl.floatTensorFactory.Create(_data: a_grad_data, _shape: ash);

            var c = a.Sigmoid();
            c.Backward(c_grad);

            for (int i = 0; i < c.Size; i++)
            {
                // multiplication is correct
                Assert.True(Math.Abs(c_expected.Data[i] - c.Data[i]) < 0.000001);
            }

            for (int i = 0; i < a_grad.Size; i++)
            {
                // a gradients are correct
                Assert.True(Math.Abs(a_grad.Data[i] - a.Grad.Data[i]) < 0.0001);    
            }
            
            
            // check that repeating doesn't break it
            c = a.Sigmoid();
            c.Backward(c_grad);

            for (int i = 0; i < c.Size; i++)
            {
                // multiplication is correct
                Assert.True(Math.Abs(c_expected.Data[i] - c.Data[i]) < 0.000001);
            }

            for (int i = 0; i < a_grad.Size; i++)
            {
                // a gradients are correct
                Assert.True(Math.Abs(a_grad.Data[i] - a.Grad.Data[i]) < 0.0001);    
            }

            // see if it's allocating new tensors during the forward pass
            ctrl.allow_new_tensors = false;
            
            // check that repeating the forward pass doesn't break it
            c = a.Sigmoid();
            c = a.Sigmoid();
            c.Backward(c_grad);

            for (int i = 0; i < c.Size; i++)
            {
                // multiplication is correct
                Assert.True(Math.Abs(c_expected.Data[i] - c.Data[i]) < 0.000001);
            }

            for (int i = 0; i < a_grad.Size; i++)
            {
                // a gradients are correct
                Assert.True(Math.Abs(a_grad.Data[i] - a.Grad.Data[i]) < 0.0001);    
            }
            ctrl.allow_new_tensors = true;
              
        }

        [Test]
        public void OneLayerMLPAutograd()
        {
            int[] input_shape = new int[] {4, 3};
            float[] input_data = new float[] { 0,  0,  1,  0,  1,  1,  1,  0,  1,  1,  1,  1};
            var input = ctrl.floatTensorFactory.Create(_data: input_data, _shape: input_shape);
            input.Autograd = true;
            
            int[] target_shape = new int[] {4, 1};
            float[] target_data = new float[] { 0,0,1,1,};
            var target = ctrl.floatTensorFactory.Create(_data: target_data, _shape: target_shape);
            target.Autograd = true;
            
            int[] grad_shape = new int[] {4, 1};
            float[] grad_data = new float[] { 1,1,1,1};
            var grad = ctrl.floatTensorFactory.Create(_data: grad_data, _shape: grad_shape);
            grad.Autograd = false;
            
            int[] weights_shape = new int[] {3, 1};
            float[] weights_data = new float[] { 0.2f,0.1f,0.3f};
            var weights = ctrl.floatTensorFactory.Create(_data: weights_data, _shape: weights_shape);
            weights.Autograd = true;

            var layer_1 = input.MM(weights).Sigmoid();
            var loss = (layer_1.Sub(target)).Pow(2);
            loss.Backward(grad);
            
            Assert.True(Math.Abs(weights.Grad.Data[0] - (-0.3395834)) < 0.0001);
            Assert.True(Math.Abs(weights.Grad.Data[1] - (0.1255458)) < 0.0001);
            Assert.True(Math.Abs(weights.Grad.Data[2] - (0.2289533)) < 0.0001);

            weights.Sub(weights.Grad, inline: true);

            ctrl.allow_new_tensors = false;
            
            layer_1 = input.MM(weights).Sigmoid();
            loss = (layer_1.Sub(target)).Pow(2);
            loss.Backward(grad);
            
            Assert.True(Math.Abs(weights.Grad.Data[0] - (-0.3249292)) < 0.0001);
            Assert.True(Math.Abs(weights.Grad.Data[1] - (0.09114857)) < 0.0001);
            Assert.True(Math.Abs(weights.Grad.Data[2] - (0.1891759)) < 0.0001);
            
            ctrl.allow_new_tensors = true;

        }


        [Test]
        public void TwoLayerMLPAutograd()
        {
            
            int[] input_shape = new int[] {4, 3};
            float[] input_data = new float[] { 0,  0,  1,  0,  1,  1,  1,  0,  1,  1,  1,  1};
            var input = ctrl.floatTensorFactory.Create(_data: input_data, _shape: input_shape);
            input.Autograd = true;
            
            int[] target_shape = new int[] {4, 1};
            float[] target_data = new float[] { 0,0,1,1,};
            var target = ctrl.floatTensorFactory.Create(_data: target_data, _shape: target_shape);
            target.Autograd = true;
            
            int[] grad_shape = new int[] {4, 1};
            float[] grad_data = new float[] { 1,1,1,1};
            var grad = ctrl.floatTensorFactory.Create(_data: grad_data, _shape: grad_shape);
            grad.Autograd = false;
            
            int[] weights1_shape = new int[] {3, 4};
            float[] weights1_data = new float[] { 0.4170f,  0.7203f, 0.0001f,  0.3023f, 0.1468f,  0.0923f,  0.1863f, 
                0.3456f, 0.3968f,  0.5388f,  0.4192f,  0.6852f};
            var weights1 = ctrl.floatTensorFactory.Create(_data: weights1_data, _shape: weights1_shape);
            weights1.Autograd = true;
            
            int[] weights2_shape = new int[] {4, 1};
            float[] weights2_data = new float[] { 0.2045f,0.8781f,0.0274f,0.6705f};
            var weights2 = ctrl.floatTensorFactory.Create(_data: weights2_data, _shape: weights2_shape);
            weights2.Autograd = true;

            var layer_1 = input.MM(weights1).Sigmoid();
            var layer_2 = layer_1.MM(weights2).Sigmoid();
            var loss = layer_2.Sub(target).Pow(2);
            loss.Backward(grad);

            Assert.True(Math.Abs(weights1.Grad.Data[0] - (-0.00559968)) < 0.00001);
            Assert.True(Math.Abs(weights1.Grad.Data[1] - (-0.01954043)) < 0.00001);
            Assert.True(Math.Abs(weights1.Grad.Data[2] - (-0.00084936)) < 0.00001);
            Assert.True(Math.Abs(weights1.Grad.Data[3] - (-0.01617729)) < 0.00001);
            
            Assert.True(Math.Abs(weights1.Grad.Data[8] - (0.02101371)) < 0.00001);
            Assert.True(Math.Abs(weights1.Grad.Data[9] - (0.09150549)) < 0.00001);
            Assert.True(Math.Abs(weights1.Grad.Data[10] - (0.00267767)) < 0.00001);
            Assert.True(Math.Abs(weights1.Grad.Data[11] - (0.06076522)) < 0.00001);

            weights1.Sub(weights1.Grad,inline:true);
            weights2.Sub(weights2.Grad,inline:true);
            
            ctrl.allow_new_tensors = false;
            
            layer_1 = input.MM(weights1).Sigmoid();
            layer_2 = layer_1.MM(weights2).Sigmoid();
            loss = layer_2.Sub(target).Pow(2);
            loss.Backward(grad);
            
            Assert.True(Math.Abs(weights1.Grad.Data[0] - (0.00273491)) < 0.00001);
            Assert.True(Math.Abs(weights1.Grad.Data[1] - (-0.036114253)) < 0.00001);
            Assert.True(Math.Abs(weights1.Grad.Data[2] - (0.01777665)) < 0.00001);
            Assert.True(Math.Abs(weights1.Grad.Data[3] - (-0.02339926)) < 0.00001);
            
            Assert.True(Math.Abs(weights1.Grad.Data[8] - (-0.00287705)) < 0.00001);
            Assert.True(Math.Abs(weights1.Grad.Data[9] - (0.05070414)) < 0.00001);
            Assert.True(Math.Abs(weights1.Grad.Data[10] - (-0.01417091)) < 0.00001);
            Assert.True(Math.Abs(weights1.Grad.Data[11] - (0.02481702)) < 0.00001);
            
            ctrl.allow_new_tensors = true;

        }
    }
}