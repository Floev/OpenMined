using System.Runtime.InteropServices;
using OpenMined.Syft.Tensor;
using UnityEngine;
using OpenMined.Network.Controllers;

namespace OpenMined.Syft.Layer.Loss
{
    public class MSELoss:Loss
    {
        public MSELoss (SyftController controller)
        {
            init("mseloss");

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);

        }
        protected override FloatTensor Forward(FloatTensor prediction, FloatTensor target)
        {
            FloatTensor output = ((prediction.Sub(target)).Pow(2)).Sum();
            return output;
        }
    }
}
