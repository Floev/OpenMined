using UnityEngine;
using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using OpenMined.Network.Utils;
using OpenMined.Network.Servers;
using OpenMined.Syft.Layer;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using OpenMined.UI;
using OpenMined.Network.Servers.BlockChain;
using System.Threading.Tasks;
using OpenMined.Network.Servers.BlockChain.Requests;
using OpenMined.Network.Servers.Ipfs;
using OpenMined.Syft.Layer.Loss;
using OpenMined.Syft.Optim;

namespace OpenMined.Network.Controllers
{
    public class Grid
    {

        private SyftController controller;
        private List<String> experiments;

        public Grid(SyftController controller)
        {
            this.controller = controller;
            experiments = new List<string>();
        }

        private async Task<bool> CheckModelFromJob(string job)
        {
            var getResultRequest = new GetResultsRequest(job);
            getResultRequest.RunRequestSync();
            var responseHash = getResultRequest.GetResponse().resultAddress;

            return responseHash != "";
        }

        private async Task<int> LoadModelFromJob(string job)
        {
            var getResultRequest = new GetResultsRequest(job);
            getResultRequest.RunRequestSync();
            var responseHash = getResultRequest.GetResponse().resultAddress;

            while (responseHash == "")
            {
                Debug.Log(string.Format("Could not load job {0}. Trying again in 1 seconds.", job));
                await Task.Delay(1000);

                // run the request again
                getResultRequest = new GetResultsRequest(job);
                getResultRequest.RunRequestSync();
                responseHash = getResultRequest.GetResponse().resultAddress;
            }

            // load the model into memory

            var response = Ipfs.Get<IpfsJob>(responseHash);
            var modelDefinition = response.Model;
            var model = this.CreateSequential(modelDefinition);

            return model.Id;
        }

        public async void GetResults(string experimentId, Action<string> response)
        {
            var experiment = Ipfs.Get<IpfsExperiment>(experimentId);
            var results = new int[experiment.jobs.Count()];
            for (var i = 0; i < experiment.jobs.Count(); ++i)
            {
                results[i] = await LoadModelFromJob(experiment.jobs[i]);
            }

            response(JsonConvert.SerializeObject(results));
            return;
        }

        public async void CheckStatus(string experimentId, Action<string> response)
        {
            var experiment = Ipfs.Get<IpfsExperiment>(experimentId);
            var results = new bool[experiment.jobs.Count()];
            for (var i = 0; i < experiment.jobs.Count(); ++i)
            {
                results[i] = await CheckModelFromJob(experiment.jobs[i]);
            }

            var allLoaded = true;
            int not_loaded = 0;
            int loaded = 0;
            for (var i = 0; i < results.Count(); ++i)
            {
                if (!results[i])
                {
                    allLoaded = false;
                    not_loaded = not_loaded+1;
                }
                else
                {
                    loaded = loaded+1;
                }
            }

            if (allLoaded)
            {
                response("Complete");    
            }

            else 
            {
                response("In progress (" + loaded.ToString() + "/" + (loaded+not_loaded).ToString() + " models are done)");
            }


            return;
        }

        public string Run(int inputId, int targetId, List<GridConfiguration> configurations, MonoBehaviour owner)
        {
            Debug.Log("Grid.Run");

            var inputTensor = controller.floatTensorFactory.Get(inputId);
            var targetTensor = controller.floatTensorFactory.Get(targetId);

            // write the input and target tensors to Ipfs
            var inputJob = new Ipfs();
            var targetJob = new Ipfs();

            var inputIpfsResponse = inputJob.Write(inputTensor.GetConfig());
            var targetIpfsResponse = targetJob.Write(targetTensor.GetConfig());

            Debug.Log("Input Hash: " + inputIpfsResponse.Hash);
            Debug.Log("Target Hash: " + targetIpfsResponse.Hash);

            var jobs = new string[configurations.Count];

            for (var i = 0; i < configurations.Count; ++i)
            {
                var config = configurations[i];
                var model = controller.GetModel(config.model) as Sequential;
                var serializedModel = model.GetConfig();

                var configJob = new Ipfs();
                var ipfsJobConfig = new IpfsJobConfig(config.lr, config.criterion, config.iters);

                var response = configJob.Write(new IpfsJob(inputIpfsResponse.Hash, targetIpfsResponse.Hash, serializedModel, ipfsJobConfig));

                jobs[i] = response.Hash;
            }

            var experiment = new IpfsExperiment(jobs);
            var experimentWriteJob = new Ipfs();
            var experimentResult = experimentWriteJob.Write(experiment);

            BlockChain chain = Camera.main.GetComponent<BlockChain>();
            owner.StartCoroutine(chain.AddExperiment(experimentResult.Hash, jobs));
            experiments.Add(experimentResult.Hash);

            return experimentResult.Hash;
        }

        public string TrainModel(IpfsJob job)
        {
            var tmpInput = Ipfs.Get<JToken>(job.input);
            var tmpTarget = Ipfs.Get<JToken>(job.target);

            var seq = CreateSequential(job.Model);

            var inputData = tmpInput.SelectToken("data").ToObject<float[]>();
            var inputShape = tmpInput.SelectToken("shape").ToObject<int[]>();
            var inputTensor = controller.floatTensorFactory.Create(_data: inputData, _shape: inputShape, _autograd: true);

            var targetData = tmpTarget.SelectToken("data").ToObject<float[]>();
            var targetShape = tmpTarget.SelectToken("shape").ToObject<int[]>();
            var targetTensor = controller.floatTensorFactory.Create(_data: targetData, _shape: targetShape, _autograd: true);

            var grad = controller.floatTensorFactory.Create(_data: new float[] { 1, 1, 1, 1 }, 
                                                            _shape: new int[] { 4, 1 });

            Loss loss;

            switch (job.config.criterion)
            {
                case "mseloss":
                    loss = new MSELoss(this.controller);
                    break;
                case "categorical_crossentropy":
                    loss = new CategoricalCrossEntropyLoss(this.controller);
                    break;
                case "cross_entropy_loss":
                    loss = new CrossEntropyLoss(this.controller, 1); // TODO -- real value
                    break;
                case "nll_loss":
                    loss = new NLLLoss(this.controller);
                    break;
                default:
                    loss = new MSELoss(this.controller);
                    break;
            }

            var optimizer = new SGD(this.controller, seq.getParameters(), job.config.lr, 0, 0);

            for (var i = 0; i < job.config.iters; ++i) {

                var pred = seq.Forward(inputTensor);
                var l = loss.Forward(pred, targetTensor);
                l.Backward();

                // TODO -- better batch size
                optimizer.Step(100, i);
            }

            var resultJob = new Ipfs();
            var response = resultJob.Write(new IpfsJob(job.input, job.target, seq.GetConfig(), job.config));

            return response.Hash;
        }

        public Sequential CreateSequential(JToken model)
        {  
            var seq = new Sequential(controller);

            var layers = model.SelectToken("config").Children();
            foreach(var layer in layers)
            {
                var layerType = layer.SelectToken("class_name");
                switch (layerType.Value<String>())
                {
                    case "Linear":
                        // weight float tensor
                        var weightData = layer.SelectToken("config.weights.data").ToObject<float[]>();

                        Linear linear = null;
                        if (layer.SelectToken("config.bias") == null)
                        {
                            var biasData = layer.SelectToken("config.bias.data").ToObject<float[]>();

                            var input = layer.SelectToken("config.input").ToObject<int>();
                            var output = layer.SelectToken("config.output").ToObject<int>();

                            linear = new Linear(controller, input: input, output: output, weights: weightData, bias: biasData);
                        }
                        else
                        {
                            var input = layer.SelectToken("config.input").ToObject<int>();
                            var output = layer.SelectToken("config.output").ToObject<int>();

                            linear = new Linear(controller, input: input, output: output, weights: weightData);
                        }
                        seq.AddLayer(linear);
                        break;
                    case "ReLU":
                        seq.AddLayer(new ReLU(controller));
                        break;
                    case "Log":
                        seq.AddLayer(new OpenMined.Syft.Layer.Log(controller));
                        break;
                    case "Dropout":
                        var rate = layer.SelectToken("config.rate").ToObject<float>();
                        var dropout = new Dropout(controller, rate);
                        seq.AddLayer(dropout);
                        break;
                    case "Softmax":
                        var dim = layer.SelectToken("config.dim").ToObject<int>();
                        seq.AddLayer(new Softmax(controller, dim));
                        break;
                    case "Sigmoid":
                        seq.AddLayer(new Sigmoid(controller));
                        break;
                }
            }

            return seq;
        }
    }

    public interface LayerDefinition {
        string GetLayerDefinition();
    }

    public class IpfsExperiment
    {
        public string[] jobs;

        public IpfsExperiment (string[] jobs)
        {
            this.jobs = jobs;
        }
    }

    public class IpfsJob
    {
        public string input;
        public string target;
        public JToken Model;
        public IpfsJobConfig config;

        public IpfsJob (string input, string target, JToken model, IpfsJobConfig config)
        {
            this.input = input;
            this.target = target;
            this.Model = model;
            this.config = config;
        }
    }

    public class IpfsJobConfig
    {
        [SerializeField] public float lr;

        public IpfsJobConfig(float lr)
        {
            this.lr = lr;
        }
    }
}