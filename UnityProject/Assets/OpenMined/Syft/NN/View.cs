using System;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer
{
    public class View : Layer
    {
        private int[] _outShape;

        public View(SyftController controller, int[] outShape)
        {
            init("view");
            _outShape = new int[outShape.Length];
            outShape.CopyTo(_outShape, 0);

#pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public override FloatTensor Forward(FloatTensor input)
        {
            return input.View(_outShape);
        }

        public override int getParameterCount(){return 0;}
    }
}
