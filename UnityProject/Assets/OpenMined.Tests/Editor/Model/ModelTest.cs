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
            float[] inputData = new float[] { 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18 };
            int[] inputShape = new int[] { 2,3,3 };
            var input = new Syft.Tensor.FloatTensor(_controller: ctrl, _data: inputData, _shape: inputShape);

            int[] kernel = new int[] { 3, 3 };
            var conv = new Conv2d(ctrl, 2, 2, kernel);
            var output = conv.Forward(input);
//            Debug.Log(output.Print());
        }
    }
}
