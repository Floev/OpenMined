using System.Runtime.InteropServices;
using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public abstract partial class BaseTensor<T>
    {
        protected bool dataOnGpu;

        protected ComputeBuffer dataBuffer;
        protected ComputeBuffer shapeBuffer;
        protected ComputeBuffer stridesBuffer;

        public bool DataOnGpu => dataOnGpu;

        public ComputeBuffer DataBuffer
        {
            get { return dataBuffer; }
            set { dataBuffer = value; }
        }

        public ComputeBuffer ShapeBuffer
        {
            get { return shapeBuffer; }
            set { shapeBuffer = value; }
        }

        public ComputeBuffer StridesBuffer
        {
            get { return stridesBuffer; }
            set { stridesBuffer = value; }
        }

        public bool Gpu(ComputeShader _shader)
        {
            if (dataOnGpu || !SystemInfo.supportsComputeShaders) return false;
            shader = _shader;
            CopyCputoGpu();
            EraseCpu();
            return true;
        }

        public void Cpu()
        {
            if (!dataOnGpu) return;
            CopyGpuToCpu();
            EraseGpu();
        }

        protected void CopyGpuToCpu()
        {
            data = new T[size];
            dataBuffer.GetData(Data);
        }

        protected void CopyCputoGpu()
        {
            dataBuffer = new ComputeBuffer(size, Marshal.SizeOf(default(T)));
            shapeBuffer = new ComputeBuffer(Shape.Length, sizeof(int));
            stridesBuffer = new ComputeBuffer(Strides.Length, sizeof(int));

            //Debug.LogFormat("Copying CPU to GPU");
//            Debug.LogFormat("Data: {0}", string.Join(",", Data));
            //Debug.LogFormat("Shape: {0}", string.Join(",", Shape));
            //Debug.LogFormat("Strides: {0}", string.Join(",", Strides));

            if( data != null )
                dataBuffer.SetData(Data);
            shapeBuffer.SetData(Shape);
            stridesBuffer.SetData(Strides);

            dataOnGpu = true;
        }

        protected void EraseCpu()
        {
            data = null;
        }

        protected void EraseGpu()
        {
            dataBuffer?.Release();
            shapeBuffer?.Release();
            stridesBuffer?.Release();
            dataOnGpu = false;
        }
    }
}
