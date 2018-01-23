using NUnit.Framework;
using OpenMined.Network.Controllers;

namespace OpenMined.Tests.Model
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
