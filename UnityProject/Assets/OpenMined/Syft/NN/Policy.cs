using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using OpenMined.Syft.Tensor;

namespace OpenMined.Syft.Layer
{
    public class Policy:Layer
    {
        private Layer model;
        
        public Policy(SyftController _controller, Layer _model)
        {
            init("policy");
            controller = _controller;
            
            model = _model;
            
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);

        }

        public override FloatTensor Forward(FloatTensor input)
        {
            return model.Forward(input);
        }

        public IntTensor Sample(FloatTensor input, int dim=1)
        {
            return Forward(input).Sample(dim);
        }
        
        protected override string ProcessMessageAsLayerObject(Command msgObj, SyftController ctrl)
        {
            switch (msgObj.functionCall)
            {
                case "sample":
                {
                    var input = ctrl.floatTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Sample(input);
                    return result.Id + "";
                }
                default: 
                {
                    return "Policy.processMessage not Implemented:" + msgObj.functionCall;
                }
            }
        }
        
        public override int getParameterCount(){return model.getParameterCount();}
    }
}