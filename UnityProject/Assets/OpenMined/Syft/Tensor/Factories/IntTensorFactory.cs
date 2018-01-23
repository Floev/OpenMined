using System.Collections.Generic;
using UnityEngine;
using OpenMined.Network.Controllers;
using System;

namespace OpenMined.Syft.Tensor.Factories
{
    
    public class IntTensorFactory
    {
        private Dictionary<int, IntTensor> tensors;
        private ComputeShader shader;
        public SyftController ctrl;

        public IntTensorFactory(ComputeShader _shader, SyftController _ctrl)
        {
            shader = _shader;
            ctrl = _ctrl; 
            tensors = new Dictionary<int, IntTensor>();
        }
        
        public IntTensor Get(int id)
        {
            return tensors[id];
        }

        public int Count()
        {
            return tensors.Count;
        }
        
        public void Delete(int id)
        {
            //Debug.LogFormat("<color=purple>Removing Tensor {0}</color>", id);

            var tensor = tensors [id];
           
            
            tensors.Remove (id);
            tensor.Dispose();

        }

        public IntTensor Create(int[] _shape,
            int[] _data = null,
            ComputeBuffer _dataBuffer = null,
            ComputeBuffer _shapeBuffer = null,
            ComputeBuffer _stridesBuffer = null,
            ComputeShader _shader = null,
            bool _copyData = true,
            bool _dataOnGpu = false,
            bool _autograd = false,
            bool _keepgrads = false,
            string _creation_op = null)
        {
            // leave this IF statement - it is used for testing.
            if (ctrl.allow_new_tensors)
            {
                IntTensor tensor = new IntTensor();

                tensor.init(this,
                    _shape,
                    _data,
                    _dataBuffer,
                    _shapeBuffer,
                    shader,
                    _copyData,
                    _dataOnGpu,
                    _autograd,
                    _keepgrads,
                    _creation_op);

                tensors.Add(tensor.Id, tensor);

                return tensor;
            }
            else
            {
                throw new Exception("Attempted to Create a new IntTensor");
            }

            
        }
       
        public ComputeShader GetShader()
        {
            return shader;
        }
    }
}