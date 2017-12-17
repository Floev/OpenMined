using JetBrains.Annotations;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using OpenMined.Syft.Tensor;
using UnityEngine;

namespace OpenMined.Syft.Model
{
    public class Sequential: Layer.Model
    {
		
        public Sequential (SyftController _controller)
        {
            init("sequential");

            this.controller = _controller;
            
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public void AddModel(Layer.Model model)
        {
            this.models.Add(model.Id);
        }

        public override FloatTensor Forward(FloatTensor input)
        {
            
            for (int i = 0; i < this.models.Count; i++)
            {
                if (i == 0)    
                {
                    input = controller.getModel(this.models[0]).Forward(input);
                }
                else
                {
                    input = controller.getModel((this.models[i])).Forward(input);
                }
            }
            return input;
        }
        
        public override string ProcessMessageLocal(Command msgObj, SyftController ctrl)
        {
            switch (msgObj.functionCall)
            {
                case "add":
                {
                    var input = ctrl.getModel(int.Parse(msgObj.tensorIndexParams[0]));
                    Debug.LogFormat("<color=magenta>Model Added to Sequential:</color> {0}", input.Id);                    
                    this.AddModel(input);
                    return input.Id + "";
                }
            }
            return "Model.processMessage not Implemented:" + msgObj.functionCall;
        }

    }
}

