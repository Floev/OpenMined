using System;
using UnityEngine;
using OpenMined.Network.Utils;
using OpenMined.Network.Controllers;
using System.Collections.Generic;
using OpenMined.Syft.Tensor.Factories;
using System.Linq;
using System.Threading.Tasks;
using OpenMined.Protobuf.Onnx;

namespace OpenMined.Syft.Tensor
{
    public partial class IntTensor : BaseTensor<int>
    {
        private string creation_op;
        public List<int> children_indices; // children -> counts
        public List<int> children_counts; // children -> counts

        private IntTensorFactory factory;

        public IntTensor()
        {
            // DON'T USE THIS CONSTRUCTOR - USE FACTORY INSTEAD.
            // factory.Create(all, my, params)
        }

        public void Init(IntTensorFactory _factory,
            int[] _shape,
            int[] _data = null,
            ComputeBuffer _dataBuffer = null,
            ComputeBuffer _shapeBuffer = null,
            ComputeBuffer _stridesBuffer = null,
            ComputeShader _shader = null,
            bool _copyData = true,
            bool _dataOnGpu = false,
            string _creation_op = null)
        {

            factory = _factory;
            dataOnGpu = _dataOnGpu;
            creation_op = _creation_op;

            // First: check that shape is valid.
            if (_shape == null || _shape.Length == 0)
            {
                throw new InvalidOperationException("Tensor shape can't be an empty array.");
            }

            // Second: since shape is valid, let's save it
            shape = (int[]) _shape.Clone();

            setStridesAndCheckShape();

            // Third: let's see what kind of data we've got. We should either have
            // a GPU ComputeBuffer or a data[] object.
            if (_data != null && _shapeBuffer == null && _stridesBuffer == null && _dataBuffer == null)
            {
                InitCpu(_data: _data, _copyData: _copyData);
            }
            else if (_dataBuffer != null && _shapeBuffer != null && _stridesBuffer != null && SystemInfo.supportsComputeShaders && _data == null)
            {
                // looks like we have GPU data being passed in... initialize a GPU tensor.

                InitGpu(_shader, _dataBuffer, _shapeBuffer, _stridesBuffer, _copyData);
            }
            else
            {
                // no data seems to be passed in... or its got missing stuff

                // if CPU works... go with that
                if (_data != null)
                {
                    InitCpu(_data, _copyData);
                }
                else if (_dataBuffer != null && _shader != null)
                {
                    if (SystemInfo.supportsComputeShaders)
                    {
                        // seems i'm just missing a shape buffer - no biggie
                        shapeBuffer = new ComputeBuffer(shape.Length, sizeof(int));
                        shapeBuffer.SetData(shape);

                        InitGpu(_shader, _dataBuffer, _shapeBuffer, _stridesBuffer, _copyData);
                        InitShaderKernels();
                    }
                    else
                    {
                        throw new InvalidOperationException(
                            "You seem to be trying to create a GPU tensor without having access to a GPU...");
                    }
                }
                else
                {
                    // nothing else seems to work - i suppose i'm just supposed to initialize an empty tensor.
                    long acc = 1;
                    for (var i = shape.Length - 1; i >= 0; --i)
                    {
                        acc *= shape[i];
                    }

                    if (_dataOnGpu)
                    {
                        _shapeBuffer = new ComputeBuffer(shape.Length, sizeof(int));
                        _shapeBuffer.SetData(shape);
                        _stridesBuffer = new ComputeBuffer(shape.Length, sizeof(int));
                        _stridesBuffer.SetData(strides);
                        _dataBuffer = new ComputeBuffer(size, sizeof(float));

                        InitGpu(_shader: _shader, _dataBuffer: _dataBuffer, _shapeBuffer: _shapeBuffer, _stridesBuffer: _stridesBuffer,
                            _copyData: false);
                        InitShaderKernels();
                        this.Zero_();
                    }
                    else
                    {
                        _data = new int[acc];

                        InitCpu(_data, false);
                    }
                }
            }

            // Lastly: let's set the ID of the tensor.
            // IDEs might show a warning, but ref and volatile seems to be working with Interlocked API.

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);

            if (SystemInfo.supportsComputeShaders && shader == null)
            {
                shader = factory.GetShader();
                InitShaderKernels();
            }
        }
        public void InitShaderKernels()
        {
            //AddElemIntKernel = this.shader.FindKernel("AddElemInt");
//            NegateKernel = this.shader.FindKernel("NegateInt");
        }

        public IntTensor Copy()
        {
            throw new NotImplementedException();
        }
        
        public override string ProcessMessage(Command msgObj, SyftController ctrl)
        {
            switch (msgObj.functionCall)
            {
                case "abs":
                {
                    var result = this.Abs();
                    return result.id + "";
                }
                case "abs_":
                {
                    var result = this.Abs(inline:true);
                    return result.id + "";
                }
                case "acos":
                {
                    var result = Acos();
                    return result.Id.ToString();
                }
                case "lt":
                {
                    var compareToTensor = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Lt(compareToTensor);
                    return result.id + "";
                }
                case "lt_":
                {
                    var compareToTensor = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Lt(compareToTensor, inline: true);
                    return this.id + "";
                }

                case "max_":
                {
                    var compareToTensor = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Max(compareToTensor, inline: true);
                    return this.id + "";
                }

                case "add_elem":
                {
                    Debug.LogFormat("add_elem");
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Add(tensor_1);
                    return result.id + "";
                }
                case "add_elem_":
                {
                    Debug.LogFormat("add_elem_");
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Add(tensor_1, inline: true);
                    return this.id + "";
                }
                case "add_scalar":
                {
                    Debug.LogFormat("add_scalar");
                    IntTensor result = this.Add(int.Parse(msgObj.tensorIndexParams[0]));
                    return result.Id + "";
                }
                case "add_scalar_":
                {
                    Debug.LogFormat("add_scalar_");
                    this.Add(int.Parse(msgObj.tensorIndexParams[0]), inline: true);
                    return this.id + "";
                }
                case "eq":
                {
                  var other = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                  var result = this.Eq(other);
                  return result.id + "";
                }
                case "eq_":
                {
                  var other = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                  this.Eq(other, inline: true);
                  return this.id + "";
                }
                case "exp":
                {
                  var result = this.Exp();
                  return result.id + "";
                }
                case "cos":
                {
                    var result = Cos();
                    return result.Id.ToString();
                } 
                case "equal":
                {
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    return Convert.ToString(this.Equal(tensor_1));
                }
                case "get":
                {
                    var param_to_get = msgObj.tensorIndexParams[0];
                    switch (param_to_get)
                    {
                        case "creation_op":
                        {
                            if (creation_op != null)
                                return creation_op;
                            return "";
                        }
                        case "data":
                        {
                            string out_str = "";

                            if (dataOnGpu)
                            {
                                int[] temp_data = new int[size];
                                dataBuffer.GetData(temp_data);
                                for (int i = 0; i < size; i++)
                                {
                                    out_str += temp_data[i] + ",";
                                }
                            }
                            else
                            {
                                for (int i = 0; i < size; i++)
                                {
                                    out_str += this[i] + ",";
                                }
                            }

                            return out_str;
                        }
                        case "dataOnGpu":
                        {
                            if (dataOnGpu)
                            {
                                return "1";
                            }
                            return "0";
                        }
                        case "id":
                        {
                            return this.id + "";
                        }
                        case "size":
                        {
                            return this.size + "";
                        }
                        case "shape":
                        {
                            string shape_str = "";
                            for (int i = 0; i < shape.Length; i++)
                            {
                                shape_str += (shape[i] + ",");
                            }
                            return shape_str;
                        }
                    }
                    return "param not found or not configured with a getter";
                }
                case "sign":
                {
                    var result = this.Sign();
                    return result.id + "";
                }
                case "pow_scalar":
                {
                    Debug.LogFormat("pow_scalar");
                    var result = this.Pow(int.Parse(msgObj.tensorIndexParams[0]));
                    return result.id + "";
                }
                case "pow_scalar_":
                {
                  Debug.LogFormat("pow_scalar_");
                  this.Pow(int.Parse(msgObj.tensorIndexParams[0]), inline: true);
                  return this.id + "";
                }

                case "pow_elem":
                {
                    Debug.LogFormat("pow_elem");
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Pow(tensor_1);
                    return result.id + "";
                }
                case "pow_elem_":
                {
                    Debug.LogFormat("pow_elem_");
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Pow(tensor_1, inline: true);
                    return this.id + "";
                }
                case "sub_elem":
                {
                    Debug.LogFormat("sub_elem");
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Sub(tensor_1);
                    return result.id + "";
                }
                case "sub_elem_":
                {
                    Debug.LogFormat("sub_elem_");
                    var tensor_1 = factory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Sub(tensor_1, inline: true);
                    return this.id + "";
                }
                case "sub_scalar":
                {
                    Debug.LogFormat("sub_scalar");
                    IntTensor result = this.Sub(int.Parse(msgObj.tensorIndexParams[0]));
                    return result.Id + "";
                }
                case "sub_scalar_":
                {
                    Debug.LogFormat("sub_scalar_");
                    this.Sub(int.Parse(msgObj.tensorIndexParams[0]), inline: true);
                    return this.id + "";
                }
                case "reciprocal":
                {
                    var result = Reciprocal();
                    return result.id.ToString();
                }
                case "reciprocal_":
                {
                    Reciprocal(inline: true);
                    return Id.ToString();
                }
                case "sqrt":
                {
                    var result = Sqrt();
                    return result.Id + "";
                }

                case "sin":
                {
                     var result = Sin();
                     return result.Id.ToString();
                }
                case "sinh":
                {
                    var result = Sinh();
                    return result.Id.ToString();
                }
				case "neg":
				{
					Debug.LogFormat("neg");
					var result = Neg();
					return result.Id.ToString();
				}

				case "neg_":
				{
					Debug.LogFormat("neg_");
					Neg(inline: true);
					return Id.ToString();
				}
                case "to_numpy":
                {
                    if (DataOnGpu)
                    {
                        var tmpData = new int[size];
                        dataBuffer.GetData(tmpData);
                        return string.Join(" ", tmpData);

                    } else
                    {
                        return string.Join(" ", Data);

                    }
                }

                case "tan":
                {
                    var result = Tan();
                    return result.Id.ToString();
                }
                case "top_k":
                {
                    Debug.LogFormat("top_k");
                    int k =int.Parse(msgObj.tensorIndexParams[0]);
                    int dim  =int.Parse(msgObj.tensorIndexParams[1]);
                    bool largest  =bool.Parse(msgObj.tensorIndexParams[2]);
                    bool sorted =bool.Parse(msgObj.tensorIndexParams[3]);
                    IntTensor result = this.TopK(k,dim,largest,sorted);
                    return result.id + "";
                }
                case "trace":
                {
                    var result = this.Trace();
                    return result.ToString();
                }
                case "transpose":
                {
                    if (msgObj.tensorIndexParams.Length != 0)
                        {
                            var dim1 = int.Parse(msgObj.tensorIndexParams[0]);
                            var dim2 = int.Parse(msgObj.tensorIndexParams[1]);
                            return Transpose(dim1, dim2).Id.ToString();
                        }
                    else
                        {
                            return Transpose().Id.ToString();
                        }
                }
                
                case "view":
                {
                    int[] new_dims = new int[msgObj.tensorIndexParams.Length];
                    for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
                    {
                        new_dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
                    }
                    var result = View(new_dims);
                    return result.Id.ToString();
                }

                case "view_":
                {
                    int[] new_dims = new int[msgObj.tensorIndexParams.Length];
                    for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
                    {
                        new_dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
                    }
                    View(new_dims, inline: true);
                    return Id.ToString();
                }

                case "unfold":
                {
                    int dim = int.Parse (msgObj.tensorIndexParams [0]);
                    int size = int.Parse (msgObj.tensorIndexParams [1]);
                    int step = int.Parse (msgObj.tensorIndexParams [2]);
                    
                    var result = Unfold (dim, size, step);
                    return result.Id.ToString ();
                }
                
                case "to_numpy_by_proto":
                {
                    return this.GetProto().ToString();
                }
            }
            return "IntTensor.processMessage: Command not found:" + msgObj.functionCall;
        }

        override protected string ToString(int var)
        {
            return var.ToString();
        }

        public TensorProto GetProto()
        {
            // TensorProto t = base.ToProto();
            int[] tmpData;
            if (DataOnGpu)
            {
                tmpData = new int[size];
                dataBuffer.GetData(tmpData);
            }
            else
            {
                tmpData = Data;
            }
            TensorProto t = new TensorProto
            {   
                Dims = { Array.ConvertAll(this.Shape, val => (long)val) },
                DataType = TensorProto.Types.DataType.Int32,
                Int32Data = { tmpData },
                Name = this.Id.ToString(),
                DocString = "",
            };

            return t;
        }

        public ValueInfoProto GetValueInfoProto ()
        {
            ValueInfoProto i = new ValueInfoProto
            {
                Name = this.Id.ToString(),
                Type = new TypeProto
                {
                    TensorType = new TypeProto.Types.Tensor
                    {
                        ElemType = TensorProto.Types.DataType.Int32,
                        Shape = new TensorShapeProto
                        {
                            Dim = { Array.ConvertAll(this.Shape, val => new TensorShapeProto.Types.Dimension { DimValue = val }) }
                        }
                    }
                }
            };

            return i;
        }
    }
}
