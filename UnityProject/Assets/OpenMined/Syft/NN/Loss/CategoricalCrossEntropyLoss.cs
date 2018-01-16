using UnityEngine;
using System.Collections;
using OpenMined.Syft.Tensor;
using OpenMined.Network.Controllers;


/**
 * Like CrossEntropyLoss, but follows the spec of Keras implementation
 * of Categorical cross entropy
 */

namespace OpenMined.Syft.Layer.Loss
{
    public class CategoricalCrossEntropyLoss : Loss
    {

        public CategoricalCrossEntropyLoss(SyftController controller)
        {
            init("categorical_crossentropy");

#pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public override FloatTensor Forward(FloatTensor prediction, FloatTensor target)
        {
            // Note: prediction should be logits, basically pre-softmax. This method applies softmax first. 
            // TODO check shapes 

            FloatTensor output = ((target.Mul(prediction.Log())).Sum()).Neg();

            return output;
        }

        public override int getParameterCount() { return 0; }

    }
}

