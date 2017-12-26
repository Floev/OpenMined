using System;
using System.Collections.Generic;
using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private bool autograd;
        public FloatTensor Grad { get; private set; }

        private bool keepgrads;

        private List<FloatTensor> creators;
        private string creation_op;
        private Dictionary<int, int> children;

        public void InitAutograd()
        {
//			if(!autograd) {
            autograd = true;
            creators = new List<FloatTensor>();
            children = new Dictionary<int, int>();
//			}
        }

        public bool AllChildrenGradsAccountedFor()
        {
            foreach (var item in children)
            {
                if (item.Value == 0)
                {
                    return false;
                }
            }
            return true;
        }

        // hook autograd many tensor parents
        public void HookAutograd(ref FloatTensor result, string creation_op, params FloatTensor[] args)
        {
            if (autograd)
            {
                result.InitAutograd();
                result.creation_op = creation_op;
                result.creators.Add(this);
                children.Add(result.Id, 0);
                foreach (FloatTensor x in args)
                {
                    result.creators.Add(x);
                    //can't pass params by ref, need to do outside
                    //x.children.Add(result.Id, 0);
                };
            }
        }

        // hook autograd two parents - one scalar
        public void HookAutograd(ref FloatTensor result, float x, string creation_op)
        {
            if (autograd)
            {
                FloatTensor new_child =
                    new FloatTensor(_controller: controller, _shape: new int[] { 1 }, _data: new float[] { x });

                result.InitAutograd();
                result.creators.Add(this);
                result.creators.Add(new_child);
                result.creation_op = creation_op;

                children.Add(result.Id, 0);
                //				new_child.children.Add (result.Id, 0);
            }
        }

        // hook autograd two parents
        public void HookAutograd(ref FloatTensor result, ref FloatTensor x, string creation_op)
        {
            if (autograd)
            {
                result.InitAutograd();
                result.creators.Add(this);
                result.creators.Add(x);
                result.creation_op = creation_op;

                children.Add(result.Id, 0);
                x.children.Add(result.Id, 0);
            }
        }

        // hook autograd single parent
        public void HookAutograd(ref FloatTensor result, string creation_op)
        {
            if (autograd)
            {
                result.InitAutograd();
                result.creators.Add(this);
                result.creation_op = creation_op;

                children.Add(result.Id, 0);
            }
        }

        public void Backward(FloatTensor grad = null, FloatTensor grad_origin = null)
        {
            if (autograd)
            {
                if (grad == null)
                {
                    grad = this.controller.createOnesTensorLike(this);
                    grad.Autograd = false;
                }

                if (grad_origin != null)
                {
                    if (children[grad_origin.Id] > 0)
                    {
                        throw new InvalidOperationException("Can't backprop more than once.");
                    }
                    else
                    {
                        children[grad_origin.Id] += 1;
                    }
                }

                if (this.Grad == null)
                {
                    this.Grad = grad;
                }
                else
                {
                    this.Grad.Add(grad, true);
                }

                // grads must not have grads of their own
                if (this.Grad.autograd == true)
                {
                    throw new InvalidOperationException("Sorry, grads cannot have grads");
                }

                // only continue backpropping if there's something to backprop into
                // only continue backpropping if all gradients (from children) are accounted for
                // override waiting for children if "backprop" was called on this variable directly
//                Debug.LogFormat("Go into graph?:{0},{1},{2},{3}", this.creators != null, this.creators.Count > 0,grad_origin == null,AllChildrenGradsAccountedFor());
                if (this.creators != null && this.creators.Count > 0 && (grad_origin == null || AllChildrenGradsAccountedFor()))
                {
//                    Debug.LogFormat("Backprop {0}", creation_op);
                    if (creation_op == "add_elem")
                    {
                        creators[0].Backward(grad.Copy(), this);
                        creators[1].Backward(grad.Copy(), this);
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
                        var toInput = grad.Conv2d(creators[1], creators[2], creators[3], creators[4], creators[5], creators[6], true);
                        //Debug.LogFormat("Conv2d input dims {0}", String.Join(",", creators[0].Shape));
                        //Debug.LogFormat("Conv2d inputUpdate dims {0}", String.Join(",", toInput.Shape));
                        creators[0].Backward(toInput, this);
                        //Debug.LogFormat("Conv2d grad dims {0}", String.Join(",", grad.Shape));
                        int[] inS = creators[0].Shape;
                        int[] viewShape = new int[]{ 1, inS[0] * inS[1], inS[2], inS[3]};
                        int[] gradS = grad.Shape;
                        int[] gradShape = new int[]{ gradS[0] * gradS[1], gradS[2], gradS[3]};
                        //Debug.LogFormat("new in/grad shape {0}/{1}", String.Join(",", viewShape), String.Join(",", gradShape));

                        FloatTensor c = creators[0].Copy();
                        c.autograd = false;
                        FloatTensor _group = new FloatTensor(controller, new int[] { 1 }, new float[] { inS[0] * gradS[0] });
                        _group.autograd = false;
                        var kernelUpdate = c.View(viewShape).Conv2d(grad.View(gradShape), creators[2], creators[3], creators[4], creators[5], _group);
                        //Debug.LogFormat("Conv2d kernel dims {0}", String.Join(",", creators[1].Shape));
                        //Debug.LogFormat("Conv2d kernelUpdate dims {0}", String.Join(",", kernelUpdate.Shape));
                        creators[1].Backward(kernelUpdate.View(creators[1].Shape), this);
                        /*
                        creators[0].Backward(grad.MM(creators[1].Transpose()), this);
                        creators[1].Backward(creators[0].Transpose().MM(grad), this);
                        */
                    }
                    else if (creation_op == "mul_elem")
				    {
					    creators[0].Backward(grad.Mul(creators[1]), this);
					    creators[1].Backward(grad.Mul(creators[0]), this);
				    }
				    else if (creation_op == "div_elem")
				    {
					    creators[0].Backward(grad.Div(creators[1]), this);
					    creators[1].Backward(grad.Div(creators[0]), this);
				    }
				    else if (creation_op == "sub_elem")
				    {
					    creators[0].Backward(grad.Copy(), this);
					    creators[1].Backward(grad.Neg(), this);
				    }
				    else if (creation_op == "mm")
				    {
                        /*
                        Debug.LogFormat("MM grad dims {0}x{1}", grad.Shape[0], grad.Shape[1]);
                        Debug.LogFormat("MM arg1 dims {0}x{1}", creators[0].Shape[0], creators[0].Shape[1]);
                        Debug.LogFormat("MM arg2 dims {0}x{1}", creators[1].Shape[0], creators[1].Shape[1]);
                        */

                        creators[0].Backward(grad.MM(creators[1].Transpose()), this);
                        creators[1].Backward(creators[0].Transpose().MM(grad), this);

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
                        creators[0].Backward(c.Neg().Add(1.0f).Mul(this).Mul(grad), this);
                    }
                    else if (creation_op == "pow_scalar")
				    {
					    FloatTensor self_nograd = creators[0].Copy();
					    self_nograd.autograd = false;
					    creators[0].Backward(self_nograd.Mul(grad).Mul(creators[1].Data[0]), this);
				    }
                    else if (creation_op == "view")
                    {
//                        Debug.LogFormat("View grad shape {0}", string.Join(",", grad.Shape));
//                        Debug.LogFormat("out shape {0}", string.Join(",",creators[0].Shape));
                        creators[0].Backward(grad.View(creators[0].Shape), this);
                    }

                    /*if (!keepgrads) {
						controller.RemoveTensor (grad.id);
					}*/
                }
		    }
	    }
    }
}