using UnityEngine;
using System;
using System.Collections.Generic;
using System.Collections;
using OpenMined.Network.Utils;
using OpenMined.Network.Servers;
using OpenMined.Syft.Layer;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using OpenMined.UI;
using OpenMined.Network.Servers.BlockChain;

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
                var model = controller.getModel(config.model) as Sequential;
                var serializedModel = model.GetConfig();

                var configJob = new Ipfs();
                var ipfsJobConfig = new IpfsJobConfig(config.lr);
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

            // 10 epochs .. make configurable
            for (var i = 0; i < 10; ++i) {
                var pred = seq.Forward(inputTensor);

                var loss = pred.Sub(targetTensor).Pow(2);
                loss.Backward(grad);

                foreach (var p in seq.getParameters())
                {
                    var pTensor = controller.floatTensorFactory.Get(p);
                    pTensor.Sub(pTensor.Grad, inline: true);
                }
            }

            var resultJob = new Ipfs();
            var config = new IpfsJobConfig(job.config.lr);
            var response = resultJob.Write(new IpfsJob(job.input, job.target, seq.GetConfig(), config));

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
                        var weightShape = layer.SelectToken("config.weights.shape").ToObject<int[]>();
                        var weightTensor = controller.floatTensorFactory.Create(_data: weightData, _shape: weightShape, _autograd: true);

                        // bias float tensor
                        var biasData = layer.SelectToken("config.bias.data").ToObject<float[]>();
                        var biasShape = layer.SelectToken("config.bias.shape").ToObject<int[]>();
                        var biasTensor = controller.floatTensorFactory.Create(_data: biasData, _shape: biasShape, _autograd: true);

                        var input = layer.SelectToken("config.input").ToObject<int>();
                        var output = layer.SelectToken("config.output").ToObject<int>();

                        var linear = new Linear(controller, input: input, output: output, weights: weightTensor, bias: biasTensor);
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
