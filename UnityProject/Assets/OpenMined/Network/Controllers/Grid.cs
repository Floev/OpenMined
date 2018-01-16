using UnityEngine;
using System;
using System.Collections.Generic;
using System.Collections;
using OpenMined.Network.Utils;
using OpenMined.Network.Servers;
using OpenMined.Syft.Tensor;
using OpenMined.Syft.Layer;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace OpenMined.Network.Controllers
{
    public class Grid
    {

        private SyftController controller;

        public Grid(SyftController controller)
        {
            this.controller = controller;
        }

        public void Run(int inputId, int targetId, List<GridConfiguration> configurations, MonoBehaviour owner)
        {
            Debug.Log("Grid.Run");

            string ipfsHash = "";

            var inputTensor = controller.floatTensorFactory.Get(inputId);
            var targetTensor = controller.floatTensorFactory.Get(targetId);

            // write the input and target tensors to Ipfs
            var inputJob = new Ipfs();
            var targetJob = new Ipfs();

            var inputIpfsResponse = inputJob.Write(inputTensor);
            var targetIpfsResponse = targetJob.Write(targetTensor);

            Debug.Log("Input Hash: " + inputIpfsResponse.Hash);
            Debug.Log("Target Hash: " + targetIpfsResponse.Hash);

            configurations.ForEach((config) => {
                var model = controller.getModel(config.model) as Sequential;
                var layers = model.getLayers();

                var serializedModel = new List<String>();

                layers.ForEach((layerId) => {
                    var layer = controller.getModel(layerId);
                    var json = layer.GetConfig().ToString(Formatting.None);

                    serializedModel.Add(json);
                });

                var configJob = new Ipfs();
                var response = configJob.Write(new IpfsModel(inputIpfsResponse.Hash, targetIpfsResponse.Hash, serializedModel, config.lr));

                ipfsHash = response.Hash;
                Debug.Log("Model Hash: " + ipfsHash);
            });

            var request = new Request();
            owner.StartCoroutine(request.AddModel(owner, ipfsHash));
        }

        public void TrainModel(IpfsModel model)
        {
            var seq = CreateSequential(model.Model);

            var tmpInput = Ipfs.Get(model.input);
            var tmpTarget = Ipfs.Get(model.target);

            var input = controller.floatTensorFactory.Create(_data: tmpInput.Data, 
                                                             _shape: tmpInput.Shape,
                                                             _autograd: true);
            var target = controller.floatTensorFactory.Create(_data: tmpTarget.Data,
                                                              _shape: tmpTarget.Shape,
                                                              _autograd: true);

            var grad = controller.floatTensorFactory.Create(_data: new float[] { 1, 1, 1, 1 }, 
                                                            _shape: new int[] { 4, 1 });

            var pred = seq.Forward(input);

            var loss = pred.Sub(target).Pow(2);
            loss.Backward(grad);

            foreach (var p in seq.getParameters())
            {
                var pTensor = controller.floatTensorFactory.Get(p);
                pTensor.Sub(pTensor.Grad, inline: true);
            }

            var layerIdxs = seq.getLayers();
            Linear lin = (Linear)controller.getModel(layerIdxs[0]);

            Debug.Log(string.Join(",", loss.Data));
        }

        private Sequential CreateSequential(List<String> model)
        {
            // TODO just assumes it is all in a seq model the seq model should probably
            // be in the JSON????      
            var seq = new Sequential(controller);

            foreach (var l in model)
            {
                var config = JObject.Parse(l);
                Layer layer = null;
                switch ((string)config["name"])
                {
                    case "linear":
                        layer = new Linear(controller, (int)config["input"], (int)config["output"]);
                        break;
                    case "softmax":
                        layer = new Softmax(controller, (int)config["dim"]);
                        break;
                    case "relu":
                        layer = new ReLU(controller);
                        break;
                    case "log":
                        layer = new Log(controller);
                        break;
                    case "dropout":
                        layer = new Dropout(controller, (float)config["rate"]);
                        break;
                }
                seq.AddLayer(layer);
            }

            return seq;
        }
    }

    public interface LayerDefinition {
        string GetLayerDefinition();
    }

    [Serializable]
    public class IpfsModel
    {
        [SerializeField] public string input;
        [SerializeField] public string target;
        [SerializeField] public List<String> Model;
        [SerializeField] public float lr;

        public IpfsModel (string input, string target, List<String> model, float lr)
        {
            this.input = input;
            this.target = target;
            this.Model = model;
            this.lr = lr;
        }
    }
}
