using System;
using System.Runtime;
using System.Collections;
using System.Collections.Generic;
using JetBrains.Annotations;
using OpenMined.Network.Controllers;
using OpenMined.Network.Utils;
using OpenMined.Syft.Tensor;
using OpenMined.Syft.Tensor.Factories;
using UnityEngine;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;


namespace OpenMined.Syft.Layer
{
    public class Sequential: Layer
    {
        // indices for layers used in forward prediction (which themselves can contain weights)
        private List<int> layers = new List<int>();
        		
        public Sequential (SyftController _controller)
        {
            init("sequential");

            this.controller = _controller;
            
            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        private int getLayer(int i)
        {
            if(i >= 0 && i < layers.Count)
                return layers[i];
            throw new ArgumentOutOfRangeException("Sub-layer " + i + " does not exist.");
        }

        public List<int> getLayers()
        {
            return this.layers;
        }

        public void AddLayer(Layer layer)
        {
            this.layers.Add(layer.Id);
        }

        public override FloatTensor Forward(FloatTensor input)
        {
            for (int i = 0; i < this.layers.Count; i++)
            {
                int layerIdx = this.layers [i];
                Layer layer = (Layer)controller.GetModel (layerIdx);

                input = layer.Forward(input);
            }
            activation = input.Id;
            return input;
        }

        protected override string ProcessParamsMessage (Command msgObj, SyftController ctrl) 
        {   
            string out_str = "";

            for (int i = 0; i < this.layers.Count; i++)
            {
                List<int> layer_params = controller.GetModel(layers[i]).getParameters();
                for (int j = 0; j < layer_params.Count; j++)
                {
                    out_str += layer_params[j].ToString() + ",";
                }
            }

            List<int> seq_params = this.getParameters ();
            for (int i = 0; i < seq_params.Count; i++)
            {
                out_str += seq_params[i].ToString() + ",";
            }

            return out_str;
        }

        protected override string ProcessMessageAsLayerObject(Command msgObj, SyftController ctrl)
        {
            switch (msgObj.functionCall)
            {
                case "add":
                {
                    // TODO: Handle adding layers better
                    var input = (Layer)ctrl.GetModel(int.Parse(msgObj.tensorIndexParams[0]));
                    Debug.LogFormat("<color=magenta>Layer Added to Sequential:</color> {0}", input.Id);                    
                    this.AddLayer(input);
                    return input.Id + "";
                }
                case "models":
                {
                    string out_str = "";

                    for (int i = 0; i < this.layers.Count; i++)
                    {

                        out_str += this.layers[i].ToString() + ",";

                    }
                    return out_str;

                }
                case "config":
                {
                    Debug.LogFormat("<color=magenta>Get Config:</color> ");

                    var config = this.GetConfig();
                    config["backend"] = "openmined";
                    
                    return config.ToString(Formatting.None);
                }
                default: 
                {
                    return "Model.processMessage not Implemented:" + msgObj.functionCall;
                }
            }
        }

        public override int getParameterCount()
        {
            int cnt = 0;
            foreach (int layer_idx in layers)
            {
                cnt += controller.GetModel(layer_idx).getParameterCount();
            }
            return cnt;
        }

        public override List<int> getParameters()
        {
            var allParams = new List<int>();
            foreach (int layer_idx in layers)
            {
                var model = controller.GetModel(layer_idx);
                foreach (int param in model.getParameters())
                {
                    allParams.Add(param);
                }
            }

            return allParams;
        }

        public override JToken GetConfig()
        {
            var _this = this;
            
            var layer_list = new JArray();
            for (int i = 0; i < this.layers.Count; i++)
            {   
                var layer = controller.GetModel(this.layers[i]);
                layer_list.Add(
                    new JObject
                    {
                        { "class_name", layer.GetType().Name },
                        { "config", layer.GetConfig() }
                    }
                );
            }

            var config = new JObject
            {
                { "class_name", _this.GetType().Name }, 
                { "config", layer_list}
            };

            return config;
        }

    }
}

