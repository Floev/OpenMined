using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using OpenMined.Syft.Tensor;
using OpenMined.Network.Utils;
using OpenMined.Syft.Layer;
using OpenMined.Syft.Model;


namespace OpenMined.Network.Controllers
{
	public class SyftController
	{
		[SerializeField] private ComputeShader shader;

		private Dictionary<int, FloatTensor> tensors;
		private Dictionary<int, Model> models;

		public SyftController (ComputeShader _shader)
		{
			shader = _shader;

			tensors = new Dictionary<int, FloatTensor> ();
			models = new Dictionary<int, Model> ();
		}

		public ComputeShader Shader {
			get { return shader; }
		}

		public float[] RandomWeights (int length)
		{
			Random.InitState (1);
			float[] syn0 = new float[length];
			for (int i = 0; i < length; i++) {
				syn0 [i] = 2 * Random.value - 1;
			}
			return syn0;
		}

		public FloatTensor getTensor (int index)
		{
			return tensors [index];
		}

		public Model getModel(int index)
		{
			return models[index];
		}

		public ComputeShader GetShader ()
		{
			return shader;
		}

		public void RemoveTensor (int index)
		{
			var tensor = tensors [index];
			tensors.Remove (index);
			tensor.Dispose ();
		}

		public int addTensor (FloatTensor tensor)
		{
			tensor.Controller = this;
			tensors.Add (tensor.Id, tensor);
			return (tensor.Id);
		}
		
		public int addModel (Model model)
		{
			models.Add (model.Id, model);
			return (model.Id);
		}

		public FloatTensor createZerosTensorLike(FloatTensor tensor) {
			FloatTensor new_tensor = tensor.Copy ();
			new_tensor.Zero_ ();
			return new_tensor;
		}

		public FloatTensor createOnesTensorLike(FloatTensor tensor) {
			FloatTensor new_tensor = tensor.Copy ();
			new_tensor.Zero_ ();
			new_tensor.Add ((float)1,true);
			return new_tensor;
		}

		public string processMessage (string json_message)
		{
			//Debug.LogFormat("<color=green>SyftController.processMessage {0}</color>", json_message);

			Command msgObj = JsonUtility.FromJson<Command> (json_message);

			switch (msgObj.objectType) {
				case "tensor":
				{
					if (msgObj.objectIndex == 0 && msgObj.functionCall == "create")
					{
						FloatTensor tensor = new FloatTensor(this, _shape: msgObj.shape, _data: msgObj.data, _shader: this.Shader);
						Debug.LogFormat("<color=magenta>createTensor:</color> {0}", string.Join(", ", tensor.Data));
						return tensor.Id.ToString();
					}
					else if (msgObj.objectIndex > tensors.Count)
					{
						return "Invalid objectIndex: " + msgObj.objectIndex;
					}
					else
					{
						FloatTensor tensor = this.getTensor(msgObj.objectIndex);
						// Process message's function
						return tensor.ProcessMessage(msgObj, this);
					}
				}
                case "model":
                {
                    if (msgObj.functionCall == "create")
                    {
                        string model_type = msgObj.tensorIndexParams[0];
                        string arg_split = "";
                        if (msgObj.tensorIndexParams.Length > 0) arg_split = " : ";

                        Debug.LogFormat("<color=magenta>createModel:</color> {0}{1}{2}", model_type, arg_split, string.Join(" ", msgObj.tensorIndexParams));

                        if (model_type == "conv2d")
                        {
                            int[] args = msgObj.tensorIndexParams.Skip(1).Select(s => int.Parse(s)).ToArray();
                            int[] kernel = { args[3], args[4] };
                            int[] stride = { args[5], args[6] };
                            int[] padding = { args[7], args[8] };
                            int[] dilation = { args[9], args[10] };
                            bool bias = args[12] != 0;
                            Conv2d model = new Conv2d(this, args[0], args[1],kernel,stride,padding,dilation,args[11], bias);
                            return model.Id.ToString();
                        }
                        if (model_type == "linear")
                        {
                            Linear model = new Linear(this, int.Parse(msgObj.tensorIndexParams[1]), int.Parse(msgObj.tensorIndexParams[2]));
                            return model.Id.ToString();
                        }
                        else if (model_type == "sequential")
                        {
                            Sequential model = new Sequential(this);
                            return model.Id.ToString();
                        }
                        else if (model_type == "sigmoid")
						{
							Sigmoid model = new Sigmoid(this);
							return model.Id.ToString();
						}
					} 
					else
					{
						Model model = this.getModel(msgObj.objectIndex);
						return model.ProcessMessage(msgObj, this);
					}
					return "hello";
				}
				default:
				break;                
			}
 
			// If not executing createTensor or tensor function, return default error.
			return "SyftController.processMessage: Command not found.";            
		}
	}
}
