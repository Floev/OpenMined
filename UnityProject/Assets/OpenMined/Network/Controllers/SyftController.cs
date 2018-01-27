using System;
using System.IO;
using System.Collections;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using OpenMined.Syft.Tensor;
using OpenMined.Network.Utils;
using OpenMined.Syft.Layer;
using OpenMined.Syft.Layer.Loss;
using OpenMined.Syft.Optim;
using OpenMined.Syft.Tensor.Factories;
using Random = UnityEngine.Random;
using OpenMined.Syft.NN.RL;
using Agent = OpenMined.Syft.NN.RL.Agent;
using OpenMined.Network.Servers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using OpenMined.Protobuf.Onnx;
using Google.Protobuf;
using OpenMined.Network.Servers.BlockChain.Requests;
using OpenMined.Network.Servers.BlockChain.Response;

namespace OpenMined.Network.Controllers
{
    public class SyftController: NetMqDelegate
	{
		[SerializeField] private ComputeShader shader;

		public FloatTensorFactory floatTensorFactory;
		public IntTensorFactory intTensorFactory;

		private Dictionary<int, Model> models;
		private Dictionary<int, Syft.NN.RL.Agent> agents;
		private Dictionary<int, Optimizer> optimizers;

		public bool allow_new_tensors = true;

        public Grid grid;

		public SyftController (ComputeShader _shader)
		{
			shader = _shader;

			floatTensorFactory = new FloatTensorFactory(_shader, this);
			intTensorFactory = new IntTensorFactory(_shader, this);

			models = new Dictionary<int, Model> ();
			agents = new Dictionary<int, Syft.NN.RL.Agent>();
			optimizers = new Dictionary<int, Optimizer>();
            grid = new Grid(this);
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

		public Model GetModel(int index)
		{
			if (models.ContainsKey(index))
			{
				return models[index];
			}
			else
			{
				return null;
			}
		}

		public Loss GetLoss(int index)
		{
			return (Loss)models[index];
		}

		public Optimizer GetOptimizer(int index)
		{
			return optimizers[index];
		}

		public ComputeShader GetShader ()
		{
			return shader;
		}

		public int AddAgent(Syft.NN.RL.Agent agent)
		{
			agents.Add (agent.Id, agent);
			return (agent.Id);
		}

		public Syft.NN.RL.Agent GetAgent(int agent_id)
		{
			if(agents.ContainsKey(agent_id))
				return agents[agent_id];
			return null;
		}

		public void SetAgentId(int old_id, int new_id)
		{
			Syft.NN.RL.Agent old = GetAgent(old_id);

			if (agents.ContainsKey(new_id))
			{
				models.Remove(new_id);
			}

			agents.Add(new_id, old);

			if (old_id != new_id)
			{
				agents.Remove(old_id);
                agents.Add(old_id, null);
			}

		}

		public int addModel (Model model)
		{
			models.Add (model.Id, model);
			return (model.Id);
		}

		public void setModelId(int old_id, int new_id)
		{
			Model old = GetModel(old_id);

			if (models.ContainsKey(new_id))
			{
				models.Remove(new_id);
			}

			models.Add(new_id, old);

			if (old_id != new_id)
			{
				models.Remove(old_id);
				models.Add(old_id, null);
			}

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

        public void ProcessMessage(string json_message, MonoBehaviour owner, Action<string> response)
		{
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

                            response(optim.Id.ToString());
                            return;
						}
						else
						{
							Optimizer optim = this.GetOptimizer(msgObj.objectIndex);
                            response(optim.ProcessMessage(msgObj, this));

                            return;
						}
					}
					case "FloatTensor":
					{
						if (msgObj.objectIndex == 0 && msgObj.functionCall == "create")
						{
							FloatTensor tensor = floatTensorFactory.Create(_shape: msgObj.shape, _data: msgObj.data, _shader: this.Shader);
                            response(tensor.Id.ToString());
                            return;
						}
						else
						{
							FloatTensor tensor = floatTensorFactory.Get(msgObj.objectIndex);
                            // Process message's function
                            response(tensor.ProcessMessage(msgObj, this));
                            return;
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
							IntTensor tensor = intTensorFactory.Create(_shape: msgObj.shape, _data: data);
                            response(tensor.Id.ToString());
                            return;
						}
						else
						{
							IntTensor tensor = intTensorFactory.Get(msgObj.objectIndex);
                                // Process message's function
                                response(tensor.ProcessMessage(msgObj, this));
                                return;
						}
					}
					case "agent":
					{
						if (msgObj.functionCall == "create")
						{
							Layer model = (Layer) GetModel(int.Parse(msgObj.tensorIndexParams[0]));
							Optimizer optimizer = optimizers[int.Parse(msgObj.tensorIndexParams[1])];
                                response(new Syft.NN.RL.Agent(this, model, optimizer).Id.ToString());
                                return;
						}

						//Debug.Log("Getting Model:" + msgObj.objectIndex);
						Syft.NN.RL.Agent agent = this.GetAgent(msgObj.objectIndex);
                            response(agent.ProcessMessageLocal(msgObj, this));
                            return;


					}
					case "model":
					{
						if (msgObj.functionCall == "create")
						{
							string model_type = msgObj.tensorIndexParams[0];

							Debug.LogFormat("<color=magenta>createModel:</color> {0}", model_type);

							if (model_type == "linear")
							{
                                    response(this.BuildLinear(msgObj.tensorIndexParams).Id.ToString());
                                    return;
							}
							else if (model_type == "relu")
							{
                                    response(this.BuildReLU().Id.ToString());
                                    return;
							}
							else if (model_type == "log")
							{
                                    response(this.BuildLog().Id.ToString());
                                    return;
							}
							else if (model_type == "dropout")
							{
                                    response(this.BuildDropout(msgObj.tensorIndexParams).Id.ToString());
                                    return;
							}
							else if (model_type == "sigmoid")
							{
                                    response(this.BuildSigmoid().Id.ToString());
                                    return;
							}
							else if (model_type == "sequential")
							{
                                    response(this.BuildSequential().Id.ToString());
                                    return;
							}
							else if (model_type == "softmax")
							{
                                    response(this.BuildSoftmax(msgObj.tensorIndexParams).Id.ToString());
                                    return;
							}
							else if (model_type == "logsoftmax")
							{
                                    response(this.BuildLogSoftmax(msgObj.tensorIndexParams).Id.ToString());
                                    return;
							}
              else if (model_type == "tanh")
              {
                                    response(new Tanh(this).Id.ToString());
                                    return;
              }
              else if (model_type == "crossentropyloss")
              {
                                    response(new CrossEntropyLoss(this, int.Parse(msgObj.tensorIndexParams[1])).Id.ToString());
                                    return;
              }
              else if (model_type == "categorical_crossentropy")
              {
                                    response(new CategoricalCrossEntropyLoss(this).Id.ToString());
                                    return;

              }
							else if (model_type == "nllloss")
							{
                                    response(new NLLLoss(this).Id.ToString());
                                    return;
							}
                            else if (model_type == "mseloss")
							{
                                    response(new MSELoss(this).Id.ToString());
                                    return;
							}
                            else if (model_type == "embedding")
                            {
                                    response(new Embedding(this, int.Parse(msgObj.tensorIndexParams[1]), int.Parse(msgObj.tensorIndexParams[2])).Id.ToString());
                                    return;
                            }
							else
							{
								Debug.LogFormat("<color=red>Model Type Not Found:</color> {0}", model_type);
							}
						}
						else
						{
							//Debug.Log("Getting Model:" + msgObj.objectIndex);
							Model model = this.GetModel(msgObj.objectIndex);
                                response(model.ProcessMessage(msgObj, this));
                                return;
						}
                            response("Unity Error: SyftController.processMessage: Command not found:" + msgObj.objectType + ":" + msgObj.functionCall);
                            return;
					}
					case "controller":
					{
						if (msgObj.functionCall == "num_tensors")
						{
                                response(floatTensorFactory.Count() + "");
                                return;
						} else if (msgObj.functionCall == "num_models")
						{
                                response(models.Count + "");
                                return;
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

                                response(allow_new_tensors + "");
                                return;
						}else if (msgObj.functionCall == "load_floattensor")
						{
							FloatTensor tensor = floatTensorFactory.Create(filepath: msgObj.tensorIndexParams[0], _shader:this.Shader);
                                response(tensor.Id.ToString());
                                return;
						}
						else if (msgObj.functionCall == "set_seed")
						{
							 Random.InitState (int.Parse(msgObj.tensorIndexParams[0]));
                                response("Random seed set!");
                                return;
						}
						else if (msgObj.functionCall == "concatenate")
						{
							List<int> tensor_ids = new List<int>();
							for (int i = 1; i < msgObj.tensorIndexParams.Length; i++)
							{
								tensor_ids.Add(int.Parse(msgObj.tensorIndexParams[i]));
							}
							FloatTensor result = Functional.Concatenate(floatTensorFactory, tensor_ids, int.Parse(msgObj.tensorIndexParams[0]));
                                response(result.Id.ToString());
                                return;
						}
						else if (msgObj.functionCall == "ones")
						{
						    int[] dims = new int[msgObj.tensorIndexParams.Length];
							for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
							{
								dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
							}
							FloatTensor result = Functional.Ones(floatTensorFactory, dims);
                                response(result.Id.ToString());
                                return;
						}
						else if (msgObj.functionCall == "randn")
						{
						    int[] dims = new int[msgObj.tensorIndexParams.Length];
							for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
							{
								dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
							}
							FloatTensor result = Functional.Randn(floatTensorFactory, dims);
                                response(result.Id.ToString());
                                return;
						}
						else if (msgObj.functionCall == "random")
						{
						    int[] dims = new int[msgObj.tensorIndexParams.Length];
							for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
							{
								dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
							}
							FloatTensor result = Functional.Random(floatTensorFactory, dims);
                                response(result.Id.ToString());
                                return;
						}
						else if (msgObj.functionCall == "zeros")
						{
						    int[] dims = new int[msgObj.tensorIndexParams.Length];
							for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
							{
								dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
							}
							FloatTensor result = Functional.Zeros(floatTensorFactory, dims);
                                response(result.Id.ToString());
                                return;
						}
						else if (msgObj.functionCall == "model_from_json")
						{
							Debug.Log("Loading Model from JSON:");
							var json_str = msgObj.tensorIndexParams[0];
							var config = JObject.Parse(json_str);

							Sequential model;

							if ((string)config["class_name"] == "Sequential")
							{
								model = this.BuildSequential();
							}
							else
							{
                                    response("Unity Error: SyftController.processMessage: while Loading model, Class :" + config["class_name"] + " is not implemented");
                                    return;
							}

							for (int i = 0; i < config["config"].ToList().Count; i++)
							{
								var layer_desc = config["config"][i];
								var layer_config_desc = layer_desc["config"];

								if ((string) layer_desc["class_name"] == "Linear"){
									int previous_output_dim;

									if (i == 0)
									{
										previous_output_dim = (int) layer_config_desc["batch_input_shape"][layer_config_desc["batch_input_shape"].ToList().Count - 1];
									}
									else
									{
										previous_output_dim = (int) layer_config_desc["units"];
									}

									string[] parameters = { "linear", previous_output_dim.ToString(), layer_config_desc["units"].ToString(), "Xavier"};
									Layer layer = this.BuildLinear(parameters);
									model.AddLayer(layer);

									string activation_name = layer_config_desc["activation"].ToString();

									if (activation_name != "linear")
									{
										Layer activation;
										if (activation_name == "softmax")
										{
											parameters = new string[] { activation_name, "1" };
											activation = this.BuildSoftmax(parameters);
										}
										else if (activation_name == "relu")
										{
											activation = this.BuildReLU();
										}
										else
										{
                                                response("Unity Error: SyftController.processMessage: while Loading activations, Activation :" + activation_name + " is not implemented");
                                                return;
										}
										model.AddLayer(activation);
									}
								}
								else
								{
                                        response("Unity Error: SyftController.processMessage: while Loading layers, Layer :" + layer_desc["class_name"] + " is not implemented");
                                        return;
								}
							}

                                response(model.Id.ToString());
                                return;
						}
						else if (msgObj.functionCall == "from_proto")
						{
							Debug.Log("Loading Model from ONNX:");
							var filename = msgObj.tensorIndexParams[0];

							var input = File.OpenRead(filename);
							ModelProto modelProto = ModelProto.Parser.ParseFrom(input);

							Sequential model = this.BuildSequential();

							foreach (NodeProto node in modelProto.Graph.Node)
							{
								Layer layer;
								GraphProto g = ONNXTools.GetSubGraphFromNodeAndMainGraph(node, modelProto.Graph);
								if (node.OpType == "Gemm")
								{
									layer = new Linear(this, g);
								}
								else if (node.OpType == "Dropout")
								{
									layer = new Dropout(this, g);
								}
								else if (node.OpType == "Relu")
								{
									layer = new ReLU(this, g);
								}
								else if (node.OpType == "Softmax")
								{
									layer = new Softmax(this, g);
								}
								else
								{
									response("Unity Error: SyftController.processMessage: Layer not yet implemented for deserialization:");
                  return;
								}
								model.AddLayer(layer);
							}

							response(model.Id.ToString());
              return;
						}
						else if (msgObj.functionCall == "to_proto")
						{
							ModelProto model =  this.ToProto(msgObj.tensorIndexParams);
							string filename = msgObj.tensorIndexParams[2];
              string type = msgObj.tensorIndexParams[3];
              if (type == "json")
              {
                response(model.ToString()); 
              }
              else
              { 
                using (var output = File.Create(filename))
                {  
                 model.WriteTo(output);
                }
                response(new FileInfo(filename).FullName);
              }
              return;
						}

						response("Unity Error: SyftController.processMessage: Command not found:" + msgObj.objectType + ":" + msgObj.functionCall);
            return;
					}
                    case "Grid":
                        if (msgObj.functionCall == "learn")
                        {
                            var inputId = int.Parse(msgObj.tensorIndexParams[0]);
                            var targetId = int.Parse(msgObj.tensorIndexParams[1]);

                            response(this.grid.Run(inputId, targetId, msgObj.configurations, owner));
                            return;
                        }

                        if (msgObj.functionCall == "getResults")
                        {
                            this.grid.GetResults(msgObj.experimentId, response);
                            return;
                        }

                        // like getResults but doesn't pause to wait for results
                        // this function will return right away telling you if
                        // it knows whether or not it is done
                        if (msgObj.functionCall == "checkStatus")
                        {
                            this.grid.CheckStatus(msgObj.experimentId, response);
                            return;
                        }

                        break;
				default:
						break;
				}   
			}
			catch (Exception e)
			{
				Debug.LogFormat("<color=red>{0}</color>",e.ToString());
                response("Unity Error: " + e.ToString());
                return;
			}

            // If not executing createTensor or tensor function, return default error.

            response("Unity Error: SyftController.processMessage: Command not found:" + msgObj.objectType + ":" + msgObj.functionCall);
            return;
		}

		private ModelProto ToProto (string[] parameters)
		{
			int modelId = int.Parse(parameters[0]);
			int inputTensorId = int.Parse(parameters[1]);
			Debug.Log("<color=yellow>Serialize to ONNX, modelId:" + modelId.ToString() + ", inputTensorId:" + inputTensorId.ToString() + "</color>");

      Model model = this.GetModel(modelId);
			ModelProto m = new ModelProto
			{
			    IrVersion = 2,
			    OpsetImport = { new OperatorSetIdProto
			    {
			        Domain = "", // operator set that is defined as part of the ONNX specification
			        Version = 2
			    }},
			    ProducerName = "openmined",
			    ProducerVersion = "0.0.2",
			    Domain = "org.openmined",
			    ModelVersion = 0,
			    DocString = "",
			    Graph = model.GetProto(inputTensorId, this),
			};

			return m;
		}

		private Sequential BuildSequential()
		{
			return new Sequential(this);
		}

		private Linear BuildLinear(string[] parameters)
		{
			int input = int.Parse(parameters[1]);
			int output = int.Parse(parameters[2]);
			string initializer =  parameters[3];

			return new Linear(this, input, output, initializer);
		}

		private Dropout BuildDropout(string[] parameters)
		{
			float rate = float.Parse(parameters[1]);

			return new Dropout(this, rate);
		}

		private ReLU BuildReLU()
		{
			return new ReLU(this);
		}

		private Log BuildLog()
		{
			return new Log(this);
		}

		private Sigmoid BuildSigmoid()
		{
			return new Sigmoid(this);
		}

		private Softmax BuildSoftmax(string[] parameters)
		{
			int reduction_dim = int.Parse(parameters[1]);
			return new Softmax(this, reduction_dim);
		}

		private LogSoftmax BuildLogSoftmax(string[] parameters)
		{
			int reduction_dim = int.Parse(parameters[1]);
			return new LogSoftmax(this, reduction_dim);
		}
    }
}
