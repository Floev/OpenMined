using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using OpenMined.Network.Controllers;
using UnityEngine;
using OpenMined.Network.Utils;
using OpenMined.Protobuf.Onnx;
using OpenMined.Syft.Tensor;

namespace OpenMined.Protobuf.Onnx
{
    public abstract partial class ONNXTools
    {
        public static GraphProto GetSubGraphFromNodeAndMainGraph(NodeProto node, GraphProto mainGraph)
        {
            List<TensorProto> inits = new List<TensorProto>();
            List<ValueInfoProto> infos = new List<ValueInfoProto>();

            List<TensorProto> allInits = new List<TensorProto>(mainGraph.Initializer);
            List<ValueInfoProto> allInfos = new List<ValueInfoProto>(mainGraph.Input);

            foreach (string id in node.Input)
            {
                // If not find it returns a null value
                TensorProto init = allInits.Find(x => x.Name == id);
                // We need to cehck the mainGraph before
                // But assuming the graph is well made
                // a null value here means that it's not a tensor needed to create the operation
                if (init == null)
                {
                    continue;
                }
                inits.Add(init);
                infos.Add(allInfos.Find(x => x.Name == id));
            }
            
            GraphProto g =  new GraphProto
            {
                Node = { node },
                Initializer = { inits },
                Input = { infos },
            };

            return g;
        }

        public static AttributeProto FindAttribute(NodeProto node, string name)
        {
            List<AttributeProto> allAttr = new List<AttributeProto>(node.Attribute);   

            return allAttr.Find(x => x.Name == name);
        }

        public static int[] GetShape(TensorProto t)
        {
            long[] longTShape = new long[t.Dims.Count];
            t.Dims.CopyTo(longTShape, 0);
            int[] tShape = Array.ConvertAll(longTShape, val => (int) val);

            return tShape;
        }

        public static FloatTensor BuildFloatTensor(TensorProto t, SyftController ctrl, bool autograd=true, bool keepgrads=true){
            int[] tShape = ONNXTools.GetShape(t);

            FloatTensor tensor;
            float[] tData;
            if (t.FloatData.Count == 0)
            {
                byte[] tmpData = t.RawData.ToByteArray();
                tData = new float[tmpData.Length / 4];
                Buffer.BlockCopy(tmpData, 0, tData, 0, tmpData.Length);
            }
            else
            {
                tData = new float[t.FloatData.Count];
                t.FloatData.CopyTo(tData, 0);
            }
            tensor = ctrl.floatTensorFactory.Create(_shape: tShape, _data: tData, _autograd: autograd, _keepgrads: keepgrads);

            return tensor;
        }

        public static IntTensor BuildIntTensor(TensorProto t, SyftController ctrl)
        {
            int[] tShape = ONNXTools.GetShape(t);

            IntTensor tensor;
            int[] tData;
            if (t.Int32Data.Count == 0)
            {
                byte[] tmpData = t.RawData.ToByteArray();
                tData = new int[tmpData.Length / 4];
                Buffer.BlockCopy(tmpData, 0, tData, 0, tmpData.Length);
            }
            else
            {
                tData = new int[t.Int32Data.Count];
                t.Int32Data.CopyTo(tData, 0);
            }
            tensor = ctrl.intTensorFactory.Create(_shape: tShape, _data: tData);

            return tensor;
        }
    }
}