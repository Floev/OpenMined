using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using OpenMined.Syft.Tensor;
using OpenMined.Network.Utils;
using OpenMined.Syft.Layer;
using OpenMined.Syft.Layer.Loss;
using OpenMined.Syft.Optim;
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
		private Dictionary<int, Optimizer> optimizers;
		
		public bool allow_new_tensors = true;

		public SyftController (ComputeShader _shader)
		{
			shader = _shader;

			floatTensorFactory = new FloatTensorFactory(_shader, this);
			intTensorFactory = new IntTensorFactory(_shader);
			
			models = new Dictionary<int, Model> ();
			optimizers = new Dictionary<int, Optimizer>();
		}

		public ComputeShader Shader {
			get { return shader; }
		}

		public float[] RandomWeights (int length, int inputSize=0)
		{
           float _inputSize = (float)inputSize;
           float Xavier = (float)Math.Sqrt(1.0F / _inputSize);
           float[] syn0 = new float[length];
           
            for (int i = 0; i < length; i++)
            {
                // Use Xavier Initialization if inputSize is given
                if (inputSize>0)
                {
                    syn0 [i] = Random.Range(-Xavier, Xavier);
                }
                else
                {
                    syn0 [i] = 2 * Random.value - 1;
                }
            }
		    return syn0;
		}

		public Model getModel(int index)
		{
			return models[index];
		}

		public Loss getLoss(int index)
		{
			return (Loss)models[index];
		}

		public Optimizer getOptimizer(int index)
		{
			return optimizers[index];
		}
		
		public ComputeShader GetShader ()
		{
			return shader;
		}
		
		public int addModel (Model model)
		{
            //Debug.LogFormat("<color=green>Adding Model {0}</color>", model.Id);
			models.Add (model.Id, model);
			return (model.Id);
		}
		
		public int addOptimizer (Optimizer optim)
		{
			optimizers.Add (optim.Id, optim);
			return (optim.Id);
		}

        public void Log(string message)
		{
			Debug.LogFormat(message);
		}
		
		public string processMessage (string json_message)
		{
			//Debug.LogFormat("<color=green>SyftController.processMessage {0}</color>", json_message);

			Command msgObj = JsonUtility.FromJson<Command> (json_message);
			try
			{

				switch (msgObj.objectType)
				{
					case "Optimizer":
					{
						if (msgObj.functionCall == "create")
						{
							string optimizer_type = msgObj.tensorIndexParams[0];

							// Extract parameters
							List<int> p = new List<int>();
							for (int i = 1; i < msgObj.tensorIndexParams.Length; i++)
							{
								p.Add(int.Parse(msgObj.tensorIndexParams[i]));
							}
							List<float> hp = new List<float>();
							for (int i = 0; i < msgObj.hyperParams.Length; i++)
							{
								hp.Add(float.Parse(msgObj.hyperParams[i]));
							}
							
							Optimizer optim = null;

							if (optimizer_type == "sgd")
							{
								optim = new SGD(this, p, hp[0], hp[1], hp[2]);
							}
							else if (optimizer_type == "rmsprop")
							{
								optim = new RMSProp(this, p, hp[0], hp[1], hp[2], hp[3]);
							}
							else if (optimizer_type == "adam")
							{
								optim = new Adam(this, p, hp[0], hp[1], hp[2], hp[3], hp[4]);
							}
							
							return optim.Id.ToString();
						}
						else
						{
							Optimizer optim = this.getOptimizer(msgObj.objectIndex);
							return optim.ProcessMessage(msgObj, this);
						}
					}
					case "FloatTensor":
					{
						if (msgObj.objectIndex == 0 && msgObj.functionCall == "create")
						{
                            FloatTensor tensor = floatTensorFactory.Create(_shape: msgObj.shape, _data: msgObj.data, _shader: this.Shader);
                            string ctnd = "";
                            if (tensor.Data.Length > 20)
                                ctnd = String.Format(", ... (size {0})", tensor.Data.Length);
                            Debug.LogFormat("<color=magenta>createTensor:{0}</color> {1}{2}", tensor.Id, string.Join(", ", tensor.Data.Take(10)), ctnd);
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
                            //Debug.LogFormat("<color=magenta>Run {0} on {1}</color>", msgObj.functionCall,tensor.Id);
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

                            Debug.LogFormat("<color=magenta>createModel:</color> {0}{1}{2}", model_type, arg_split, 
                                string.Join(" ", msgObj.tensorIndexParams));

                            if (model_type == "conv2d")
                            {
                                int[] args = msgObj.tensorIndexParams.Skip(1).Select(s => int.Parse(s)).ToArray();
                                int[] kernel = { args[2], args[3] };
                                int[] stride = { args[4], args[5] };
                                int[] padding = { args[6], args[7] };
                                int[] dilation = { args[8], args[9] };
                                bool bias = args[11] != 0;
                                Conv2d model = new Conv2d(this, args[0], args[1], kernel, stride, padding, dilation, args[10], bias);
                                return model.Id.ToString();
                            }							
							if (model_type == "linear")
							{
								return new Linear(this, int.Parse(msgObj.tensorIndexParams[1]),
								int.Parse(msgObj.tensorIndexParams[2]),
								msgObj.tensorIndexParams[3]).Id.ToString();
							}
							else if (model_type == "relu")
							{
								return new ReLU(this).Id.ToString();
							}
							else if (model_type == "log")
							{
								return new Log(this).Id.ToString();
							}
							else if (model_type == "dropout")
							{
								return new Dropout(this,float.Parse(msgObj.tensorIndexParams[1])).Id.ToString();
							}
							else if (model_type == "sigmoid")
							{
								return new Sigmoid(this).Id.ToString();
							}
							else if (model_type == "sequential")
							{
								return new Sequential(this).Id.ToString();
							}
							else if (model_type == "softmax")
							{
								return new Softmax(this,int.Parse(msgObj.tensorIndexParams[1])).Id.ToString();
							}
							else if (model_type == "logsoftmax")
							{
								return new LogSoftmax(this,int.Parse(msgObj.tensorIndexParams[1])).Id.ToString();
							}
							else if (model_type == "policy")
							{
								return new Policy(this,(Layer)getModel(int.Parse(msgObj.tensorIndexParams[1]))).Id.ToString();
							}
                            else if (model_type == "tanh")
                            {
                                return new Tanh(this).Id.ToString();
                            }
                            else if (model_type == "crossentropyloss")
                            {
                                return new CrossEntropyLoss(this, int.Parse(msgObj.tensorIndexParams[1])).Id.ToString();
                            }
							else if (model_type == "nllloss")
							{
								return new NLLLoss(this).Id.ToString();
							}
                            else if (model_type == "mseloss")
							{
								return new MSELoss(this).Id.ToString();
							}
                            else if (model_type == "embedding")
                            {
                                return new Embedding(this, int.Parse(msgObj.tensorIndexParams[1]), int.Parse(msgObj.tensorIndexParams[2])).Id.ToString();
                            }
							else
							{
								Debug.LogFormat("<color=red>Model Type Not Found:</color> {0}", model_type);
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
						}
                        else if (msgObj.functionCall == "num_models")
                        {
                                return models.Count + "";
						}
                        else if (msgObj.functionCall == "new_tensors_allowed")
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
                        else if (msgObj.functionCall == "load_floattensor")
						{
							FloatTensor tensor = floatTensorFactory.Create(filepath: msgObj.tensorIndexParams[0], _shader:this.Shader);
							return tensor.Id.ToString();
						}
						else if (msgObj.functionCall == "set_seed")
						{
							 Random.InitState (int.Parse(msgObj.tensorIndexParams[0]));
                             return "Random seed set!";
						}
						else if (msgObj.functionCall == "concatenate")
						{
							List<int> tensor_ids = new List<int>();
							for (int i = 1; i < msgObj.tensorIndexParams.Length; i++)
							{
								tensor_ids.Add(int.Parse(msgObj.tensorIndexParams[i]));
							}
							FloatTensor result = Functional.Concatenate(floatTensorFactory, tensor_ids, int.Parse(msgObj.tensorIndexParams[0]));
							return result.Id.ToString();
						}
						else if (msgObj.functionCall == "ones")
						{
						    int[] dims = new int[msgObj.tensorIndexParams.Length];
							for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
							{
								dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
							}
							FloatTensor result = Functional.Ones(floatTensorFactory, dims);
							return result.Id.ToString();
						}
						else if (msgObj.functionCall == "randn")
						{
						    int[] dims = new int[msgObj.tensorIndexParams.Length];
							for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
							{
								dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
							}
							FloatTensor result = Functional.Randn(floatTensorFactory, dims);
							return result.Id.ToString();
						}
						else if (msgObj.functionCall == "random")
						{
						    int[] dims = new int[msgObj.tensorIndexParams.Length];
							for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
							{
								dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
							}
							FloatTensor result = Functional.Random(floatTensorFactory, dims);
							return result.Id.ToString();
						}
						else if (msgObj.functionCall == "zeros")
						{
						    int[] dims = new int[msgObj.tensorIndexParams.Length];
							for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
							{
								dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
							}
							FloatTensor result = Functional.Zeros(floatTensorFactory, dims);
							return result.Id.ToString();
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