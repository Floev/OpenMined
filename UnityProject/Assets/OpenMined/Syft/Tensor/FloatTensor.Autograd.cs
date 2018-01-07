using System;
using System.Collections.Generic;
using UnityEngine;
using OpenMined.Syft.NN;
using System.Linq;
using UnityEngine;
//using Vuforia;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
	    private bool autograd;
	    public FloatTensor Grad { get; private set; }
	    private bool keepgrads;
	    
	    // checks to see if a variable has accumulated all the gradients it should before it backprops
	    public bool AllAutogradChildrenAccountedFor()
	    {
		    for (int i=0; i< children_counts.Count; i++)
		    {
			    if (children_counts[i] == 0 && factory.Get(children_indices[i]).autograd)
			    {
				    return false;
			    }
		    }
		    return true;
	    }
	    
	    public void Backward(FloatTensor grad = null, FloatTensor grad_origin = null)
	    {
		    //Debug.Log("Backward:" + this.id + " creation_op:" + creation_op);
		  
		    if (autograd)
		    {
			    
			    if (grad == null)
			    {
				    Debug.Log("Grad not Found... Creating Gradient of 1s");
				    grad = this.createOnesTensorLike();
				    grad.Autograd = false;
			    }

			    if (grad_origin != null)
			    {
				    int child_index = children_indices.IndexOf(grad_origin.Id);
				    if (children_counts[child_index] > 0)
				    {
					    throw new InvalidOperationException("Can't backprop more than once.");
				    }
				    else
				    {
					    children_counts[child_index] += 1;
				    }
			    }

			    if (this.Grad == null)
			    {
				    this.Grad = grad;
				    //Debug.Log("Setting Grad Tensor Id:" + this.id);
			    }
			    else
			    {
				    if (this.Grad.id == grad.id)
				    {
					    // do nothing
					    //Debug.Log("Not Updating For Tensor Id:" + this.id);
				    }
				    else
				    {
					    //Debug.Log("Updating For Tensor Id:" + this.id);
					    //this.Grad.Zero_();
					    this.Grad.Add(grad, inline: true);
				    }

			    }

			    // grads must not have grads of their own
			    if (this.Grad.autograd == true)
			    {
				    throw new InvalidOperationException("Sorry, grads cannot have grads");
			    }
			    
				// RULES FOR AUTOGRAD:
			    // 1) if you need to use "this" for calculating a gradient, copy it first and set autograd to false (see sigmoid)
			    // 2) if you use a method in your backprop logic that doesn't hook into the dynamic graph yet, backprop
			    // will not work!!! Make sure there's a "hookautograd" function in every method you use for backprop.
			    // 3) whenever backpropping into a method where the forward prop involved a scalar (such as scalar
			    // multiplication), current implementations assume you will NOT backprop into the scalar itself.
			    // 4) Because of rule (2), do NOT use "emptyTensorCopy" at all in backprop unless you know what you're
			    // doing. 
			    // 5) I will be especially strict about Unit tests for all backprop logic as this is the most complex
			    // piece of functionality we have. Furthermore, most errors go completely undetected (not discovered
			    // by runtime errors). Autograd bugs just make convergence go slowly and sub-optimally.
			    // 6) If you use a forward propagation tensor to backprop, you MUST remember to turn off autograd
			    // when backpropagating (see "mm" below for example). Otherwise, it will cause autograd to break because
			    // whatever child you select will think it needs to wait for another gradient before backpropagating.
			    
			    // only continue backpropping if there's something to backprop into
			    // only continue backpropping if all gradients (from children) are accounted for
			    // override waiting for children if "backprop" was called on this variable directly
			    if (this.creators != null && this.creators.Count > 0 && (grad_origin == null || AllAutogradChildrenAccountedFor()))
			    {
				    
				    if (creation_op == "add_elem")
				    {

					    factory.Get(creators[0]).Backward(grad, this);
					    factory.Get(creators[1]).Backward(grad, this);

				    }
				    else if (creation_op == "add_scalar")
				    {
					    factory.Get(creators[0]).Backward(grad, this);
				    }
				    else if (creation_op == "copy")
				    {
					    factory.Get(creators[0]).Backward(grad, this);
				    }
                    else if (creation_op == "conv2d")
                    {
                        /*
                        Debug.LogFormat("Conv2d grad dims {0}", String.Join(",", grad.Shape));
                        Debug.LogFormat("Conv2d size creators {0}", creators.Count);
                        Debug.LogFormat("Conv2d arg1 dims {0}", String.Join(",", creators[0].Shape));
                        Debug.LogFormat("Conv2d arg2 dims {0}", String.Join(",", creators[1].Shape));
                        Debug.LogFormat("Conv2d arg3 dims {0}", String.Join(",", creators[2].Shape));
                        Debug.LogFormat("Conv2d arg4 dims {0}", String.Join(",", creators[3].Shape));
                        Debug.LogFormat("Conv2d arg5 dims {0}", String.Join(",", creators[4].Shape));
                        */
                        var toInput = grad.Conv2d(factory.Get(creators[1]), factory.Get(creators[2]), factory.Get(creators[3]), factory.Get(creators[4]), factory.Get(creators[5]), factory.Get(creators[6]), true);
                        //Debug.LogFormat("Conv2d input dims {0}", String.Join(",", creators[0].Shape));
                        //Debug.LogFormat("Conv2d inputUpdate dims {0}", String.Join(",", toInput.Shape));
                        factory.Get(creators[0]).Backward(toInput, this);
                        //Debug.LogFormat("Conv2d grad dims {0}", String.Join(",", grad.Shape));
                        int[] inS = factory.Get(creators[0]).Shape;
                        int[] viewShape = new int[]{ 1, inS[0] * inS[1], inS[2], inS[3]};
                        int[] gradS = grad.Shape;
                        int[] gradShape = new int[]{ gradS[0] * gradS[1], gradS[2], gradS[3]};
                        //Debug.LogFormat("new in/grad shape {0}/{1}", String.Join(",", viewShape), String.Join(",", gradShape));

                        FloatTensor c = factory.Get(creators[0]).Copy();
                        c.autograd = false;
                        FloatTensor _group = factory.Create(new int[] { 1 }, new float[] { inS[0] * gradS[0] });
                        _group.autograd = false;
                        var kernelUpdate = c.View(viewShape).Conv2d(grad.View(gradShape), factory.Get(creators[2]), factory.Get(creators[3]),
                            factory.Get(creators[4]), factory.Get(creators[5]), _group);
                        //Debug.LogFormat("Conv2d kernel dims {0}", String.Join(",", creators[1].Shape));
                        //Debug.LogFormat("Conv2d kernelUpdate dims {0}", String.Join(",", kernelUpdate.Shape));
                        factory.Get(creators[1]).Backward(kernelUpdate.View(factory.Get(creators[1]).Shape), this);
                        /*
                        creators[0].Backward(grad.MM(creators[1].Transpose()), this);
                        creators[1].Backward(creators[0].Transpose().MM(grad), this);
                        */
                    }
				    else if (creation_op == "div_elem")
				    {
					    FloatTensor x = factory.Get(creators[0]);
					    FloatTensor y = factory.Get(creators[1]);

					    x.Backward(grad.Div(y));

					    FloatTensor y2 = y.Pow(2);
					    FloatTensor xn = x.Neg();
					    FloatTensor xny2 = xn.Div(y2);
					    FloatTensor gradxny2 = grad.Mul(xny2);
					    y.Backward(gradxny2);
                    }
                    else if (creation_op == "div_scalar")
				    {
					    factory.Get(creators[0]).Backward(grad.Div(factory.Get(creators[1]).data[0]), this);
				    }
				    else if (creation_op == "mul_elem")
				    {
					    factory.Get(creators[0]).Backward(grad.Mul(factory.Get(creators[1])), this);
					    factory.Get(creators[1]).Backward(grad.Mul(factory.Get(creators[0])), this);
				    }
				    else if (creation_op == "mul_scalar")
				    {
					    factory.Get(creators[0]).Backward(grad.Mul(factory.Get(creators[1]).data[0]), this);
				    }
				    else if (creation_op == "mm")
				    {
					    FloatTensor x = factory.Get(creators[1]).Transpose();
					    x.autograd = false;

					    FloatTensor y = factory.Get(creators[0]).Transpose();
					    y.autograd = false;

                        /*
                        Debug.LogFormat("MM grad dims {0}x{1}", grad.Shape[0], grad.Shape[1]);
                        Debug.LogFormat("MM arg1 dims {0}x{1}", creators[0].Shape[0], creators[0].Shape[1]);
                        Debug.LogFormat("MM arg2 dims {0}x{1}", creators[1].Shape[0], creators[1].Shape[1]);
                        */
					    
					    factory.Get(creators[0]).Backward(grad.MM(x), this);
					    factory.Get(creators[1]).Backward(y.MM(grad), this);
				    }
				    else if (creation_op == "neg")
				    {
					    factory.Get(creators[0]).Backward(grad.Neg(), this);
				    }
				    else if (creation_op == "pow_scalar")
				    {
                        FloatTensor x = factory.Get(creators[0]).Copy();
					    x.autograd = false;
					    
					    factory.Get(creators[0]).Backward(x.Mul(grad).Mul(factory.Get(creators[1]).Data[0]), this);
				    }
				    else if (creation_op == "sub_elem")
				    {
					    factory.Get(creators[0]).Backward(grad, this);
					    factory.Get(creators[1]).Backward(grad.Neg(), this);
				    }
				    else if (creation_op == "sub_scalar")
				    {
					    factory.Get(creators[0]).Backward(grad, this);
				    }
                    else if (creation_op == "sigmoid")
                    {
                        /*
                        Debug.LogFormat("sigmoid grad shape {0}", string.Join(",", grad.Shape));
                        Debug.LogFormat("sigmoid out shape {0}", string.Join(",", creators[0].Shape));
                        Debug.LogFormat("sigmoid val shape {0}", string.Join(",", Shape));
                        */
                        FloatTensor c = this.Copy();
                        c.autograd = false;
                        factory.Get(creators[0]).Backward(c.Neg().Add(1.0f).Mul(this).Mul(grad), this);
                    }
                    else if (creation_op == "pow_scalar")
				    {
					    FloatTensor self_nograd = this.Copy();
					    self_nograd.autograd = false;
					    
					    factory.Get(creators[0]).Backward(self_nograd.Neg().Add((float) 1).Mul(self_nograd).Mul(grad), this);
				    }

 				    else if (creation_op.Contains("sum"))
 				    {
 						// TOOD: sum backprop logic   
 					    FloatTensor parent = factory.Get(creators[0]);
					    
 					    int[] view_shape = (int[])parent.shape.Clone();
 					    view_shape[int.Parse(creation_op.Split('_')[1])] = 1;

 					   	parent.Backward(grad.View(view_shape).expand(parent.shape).Contiguous());
 				    }
// 					else if (creation_op.Contains("sum-"))
// 					{
// 						FloatTensor input = factory.Get(creators[0]).Copy();
// 						input.autograd = false;
		
// 						var dim = input.Shape.Length - 1;
// 						var split = creation_op.Split('-');
// 						if (split.Length > 1)
// 						{
// 							dim = int.Parse(split[1]);
// 						}
		
// 						// right now this function only supports grads the same size as the output
// 						// and the grad must be contiguous
// 						if(grad.Shape.SequenceEqual(this.Shape) && grad.Strides.SequenceEqual(this.Strides)) {
// 							var res = SumGradient(input, grad, dim);
// 							factory.Get(creators[0]).Backward(res, this);
// 						} else {
// 							throw new InvalidOperationException("Unable to calculate grad on output of different shape or stride");
// 						}
// 					}
				    else if (creation_op == "transpose")
				    {
					    factory.Get(creators[0]).Backward(grad.Transpose());
				    }
				    else if (creation_op == "tanh")
				    {
					    FloatTensor c = this.Copy();
					    c.autograd = false;

					    factory.Get(creators[0]).Backward(c.Pow(2).Neg().Add(1f).Mul(grad), this);
				    }
				    else if (creation_op.Contains("softmax-"))
				    {

					    FloatTensor c = this.Copy();
					    c.autograd = false;
					    var dim = int.Parse(creation_op.Split('-')[1]);
					    factory.Get(creators[0]).Backward(Functional.SoftmaxGradient(this, grad, dim), this);

				    }
				    else if (creation_op == "view")
				    {
					    FloatTensor parent = factory.Get(creators[0]);
					    parent.Backward(this.Grad.View(parent.shape));
                        /*
//                        Debug.LogFormat("View grad shape {0}", string.Join(",", grad.Shape));
//                        Debug.LogFormat("out shape {0}", string.Join(",",creators[0].Shape));
                        creators[0].Backward(grad.View(creators[0].Shape), this);
                        */
				    }
				    else
				    {
					    Debug.Log("Autograd couldn't find matching operation for:" + creation_op);   
				    }
			    }
            }
            else
		    {
			    Debug.Log("Autograd off - skipping backprop...");
		    }
	    }
        
        private FloatTensor SumGradient(FloatTensor input, FloatTensor grad, int dim)
        {
            // want to make grad look like this
            var inputShape = input.Shape;
            var stride = input.Strides;

            var gradData = grad.Data;
            var newData = new List<float>();
            
            // once we have proper support for non-contiguous tensors
            // most of this code can be replaced with a view and an expand
            // view the grad to add a singleton dimension in the dimension
            // of the sum and then expand it to the size of the input

            if (dim == 0)
            {
                var st = stride[dim];
                var sh = inputShape[dim];

                for (var i = 0; i < sh; i++)
                {
                    newData.AddRange(gradData);
                }
            }
            else
            {
                var index = 0;

                var totalSize = 1;
                for (var i = 0; i < inputShape.Length; i++)
                {
                    totalSize *= inputShape[i];
                }

                for (var i = 0; i < totalSize / (inputShape[dim] * stride[dim]); i++)
                {
                    for (var j = 0; j < inputShape[dim]; j++)
                    {
                        var segment = new ArraySegment<float>(gradData, index, stride[dim]);
                        newData.AddRange(segment);
                    }

                    index += stride[dim];
                }
            }

            return factory.Create( _shape: inputShape, _data: newData.ToArray());
        }
    }
}