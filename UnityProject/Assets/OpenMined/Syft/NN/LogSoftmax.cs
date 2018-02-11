using System;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using OpenMined.Protobuf.Onnx;

namespace OpenMined.Syft.Layer
{
    public class LogSoftmax : Layer
    {

        private int dim = 0;
        
        public LogSoftmax(SyftController controller, int dim)
        {
            init("logsoftmax");

            this.dim = dim;
            
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public override FloatTensor Forward(FloatTensor input)
        {
            FloatTensor output = input.Softmax(this.dim).Log();
            activation = output.Id;

            return output;
        }
        
        public override int getParameterCount(){return 0;}

        // See https://github.com/onnx/onnx/blob/master/docs/Operators.md#LogSoftmax
        public override GraphProto GetProto(int inputTensorId, SyftController ctrl)
        {
            FloatTensor input_tensor = ctrl.floatTensorFactory.Get(inputTensorId);
            if (activation != null)
            {
                this.Forward(input_tensor);
            }

            NodeProto node = new NodeProto
            {
                Input = { inputTensorId.ToString() },
                Output = { activation.ToString() },
                OpType = "LogSoftmax",
                Attribute = { new AttributeProto{
                    Name = "axis",
                    Type = AttributeProto.Types.AttributeType.Int,
                    I = this.dim
                }}
            };

            ValueInfoProto input_info = input_tensor.GetValueInfoProto();

            GraphProto g =  new GraphProto
            {
                Name = Guid.NewGuid().ToString("N"),
                Node = { node },
                Initializer = {  },
                Input = { input_info },
                Output = { ctrl.floatTensorFactory.Get(activation).GetValueInfoProto() },
            };

            return g;            
        }
    }
}