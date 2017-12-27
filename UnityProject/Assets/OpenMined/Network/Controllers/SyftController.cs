using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using OpenMined.Syft.Tensor;
using OpenMined.Network.Utils;
using OpenMined.Syft.Layer;
using OpenMined.Syft.Layer.Loss;
using OpenMined.Syft.Tensor.Factories;
using Random = UnityEngine.Random;


namespace OpenMined.Network.Controllers
{
	public class SyftController
	{
		[SerializeField] private ComputeShader shader;

		public FloatTensorFactory floatTensorFactory;
		public IntTensorFactory intTensorFactory;
		
		private Dictionary<int, Model> models;
		
		public bool allow_new_tensors = true;

		public SyftController (ComputeShader _shader)
		{
			shader = _shader;

			floatTensorFactory = new FloatTensorFactory(_shader, this);
			intTensorFactory = new IntTensorFactory(_shader);
			
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

		public Model getModel(int index)
		{
			return models[index];
		}

		public ComputeShader GetShader ()
		{
			return shader;
		}
		
		public int addModel (Model model)
		{
			models.Add (model.Id, model);
			return (model.Id);
		}

		public string processMessage (string json_message)
		{
			//Debug.LogFormat("<color=green>SyftController.processMessage {0}</color>", json_message);

			Command msgObj = JsonUtility.FromJson<Command> (json_message);
			try
			{

				switch (msgObj.objectType)
				{
					case "FloatTensor":
					{
						if (msgObj.objectIndex == 0 && msgObj.functionCall == "create")
						{
							FloatTensor tensor = floatTensorFactory.Create(_shape: msgObj.shape, _data: msgObj.data, _shader: this.Shader);
							return tensor.Id.ToString();
						}
						else
						{
							FloatTensor tensor = floatTensorFactory.Get(msgObj.objectIndex);
							// Process message's function
							return tensor.ProcessMessage(msgObj, this);
						}
					}
					case "IntTensor":
					{
						if (msgObj.objectIndex == 0 && msgObj.functionCall == "create")
						{
							int[] data = new int[msgObj.data.Length];
							for (int i = 0; i < msgObj.data.Length; i++)
							{
								data[i] = (int)msgObj.data[i];
							}
							IntTensor tensor = intTensorFactory.Create(_shape: msgObj.shape, _data: data, _shader: this.Shader);
							return tensor.Id.ToString();
						}
						else
						{
							IntTensor tensor = intTensorFactory.Get(msgObj.objectIndex);
							// Process message's function
							return tensor.ProcessMessage(msgObj, this);
						}
					}
					case "model":
					{
						if (msgObj.functionCall == "create")
						{
							string model_type = msgObj.tensorIndexParams[0];

							if (model_type == "linear")
							{
								Debug.LogFormat("<color=magenta>createModel:</color> {0} : {1} {2}", model_type,
									msgObj.tensorIndexParams[1], msgObj.tensorIndexParams[2]);
								Linear model = new Linear(this, int.Parse(msgObj.tensorIndexParams[1]), int.Parse(msgObj.tensorIndexParams[2]));
								return model.Id.ToString();
							}
							else if (model_type == "sigmoid")
							{
								Debug.LogFormat("<color=magenta>createModel:</color> {0}", model_type);
								Sigmoid model = new Sigmoid(this);
								return model.Id.ToString();
							}
							else if (model_type == "sequential")
							{
								Debug.LogFormat("<color=magenta>createModel:</color> {0}", model_type);
								Sequential model = new Sequential(this);
								return model.Id.ToString();
							}
							else if (model_type == "policy")
							{
								Debug.LogFormat("<color=magenta>createModel:</color> {0}", model_type);
								Policy model = new Policy(this,(Layer)getModel(int.Parse(msgObj.tensorIndexParams[1])));
								return model.Id.ToString();
							}
                            else if (model_type == "tanh")
                            {
                                Debug.LogFormat("<color=magenta>createModel:</color> {0}", model_type);
                                Tanh model = new Tanh(this);
                                return model.Id.ToString();
                            }
                            else if (model_type == "crossentropyloss")
                            {
                                Debug.LogFormat("<color=magenta>createModel:</color> {0}", model_type);
                                CrossEntropyLoss model = new CrossEntropyLoss(this);
                                return model.Id.ToString();
                            }
                            else if (model_type == "mseloss")
                            {
                                Debug.LogFormat("<color=magenta>createModel:</color> {0}", model_type);
                                MSELoss model = new MSELoss(this);
                                return model.Id.ToString();
                            }

						}
						else
						{
							Model model = this.getModel(msgObj.objectIndex);
							return model.ProcessMessage(msgObj, this);
						}
                        return "Unity Error: SyftController.processMessage: Command not found:" + msgObj.objectType + ":" + msgObj.functionCall;
					}
					case "controller":
					{
						if (msgObj.functionCall == "num_tensors")
						{
							return floatTensorFactory.Count() + "";
						} else if (msgObj.functionCall == "num_models")
						{
							return models.Count + "";
						} else if (msgObj.functionCall == "new_tensors_allowed")
						{
							
							
								Debug.LogFormat("New Tensors Allowed:{0}", msgObj.tensorIndexParams[0]);	
								if (msgObj.tensorIndexParams[0] == "True")
								{
									allow_new_tensors = true;
								} else if (msgObj.tensorIndexParams[0] == "False")
								{
									allow_new_tensors = false;
								}
								else
								{
									throw new Exception("Invalid parameter for new_tensors_allowed. Did you mean true or false?");
								}
							
							return allow_new_tensors + "";
						}
						return "Unity Error: SyftController.processMessage: Command not found:" + msgObj.objectType + ":" + msgObj.functionCall;
					}
						
				default:
						break;
				}
			}
			catch (Exception e)
			{
				Debug.LogFormat("<color=red>{0}</color>",e.ToString());
				return "Unity Error: " + e.ToString();
			}

			// If not executing createTensor or tensor function, return default error.
			return "Unity Error: SyftController.processMessage: Command not found:" + msgObj.objectType + ":" + msgObj.functionCall;            
		}
	}
}