using System;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using NUnit.Framework;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Layer;
using OpenMined.Syft.Layer.Loss;
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


    }
}
