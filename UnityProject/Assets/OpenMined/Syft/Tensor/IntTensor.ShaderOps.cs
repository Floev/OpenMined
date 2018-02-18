using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class IntTensor
    {
        // kernel pointers
        [SerializeField] private static int AddElemIntKernel;
        [SerializeField] private static int SubElemIntKernel;
        [SerializeField] private static int SubElemIntKernel_;
        [SerializeField] private static int NegateKernel;
        [SerializeField] private static int ReciprocalIntKernel;
        [SerializeField] private static int ReciprocalIntKernel_;
        [SerializeField] private static int SinIntKernel;
        [SerializeField] private static int CosIntKernel;

        public IntTensor AddElemGPU(IntTensor tensor, IntTensor result)
        {
            int kernel_id = shader.FindKernel("AddElemInt");

            shader.SetBuffer(kernel_id, "AddElemIntDataA", this.DataBuffer);
            shader.SetBuffer(kernel_id, "AddElemIntDataB", tensor.DataBuffer);
            shader.SetBuffer(kernel_id, "AddElemIntDataResult", result.DataBuffer);

            shader.Dispatch(kernel_id, this.size, 1, 1);

            return result;
        }

        public void AddElemGPU_(IntTensor tensor)
        {
            int kernel_id = shader.FindKernel("AddElemInt_");

            shader.SetBuffer(kernel_id, "AddElemIntDataA_", this.DataBuffer);
            shader.SetBuffer(kernel_id, "AddElemIntDataB_", tensor.DataBuffer);

            shader.Dispatch(kernel_id, this.size, 1, 1);
        }

        public IntTensor ReciprocalGPU(IntTensor result)
        {            
            int kernel_id = shader.FindKernel("ReciprocalInt");

            shader.SetBuffer(kernel_id, "ReciprocalIntData", this.DataBuffer);
            shader.SetBuffer(kernel_id, "ReciprocalIntDataResult", result.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);
            return result;
        }

        public void ReciprocalGPU_()
        {
            int kernel_id = shader.FindKernel("ReciprocalInt_");
            shader.SetBuffer(kernel_id, "ReciprocalIntData_", this.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);
        }

        public FloatTensor SinGPU(FloatTensor result)
        {            
            int kernel_id = shader.FindKernel("SinInt");

            shader.SetBuffer(kernel_id, "SinIntData", this.DataBuffer);
            shader.SetBuffer(kernel_id, "SinIntDataResult", result.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);
            return result;
        }

        public FloatTensor CosGPU(FloatTensor result)
        {
            int kernel_id = shader.FindKernel("CosInt");

            shader.SetBuffer(kernel_id, "CosIntData", this.DataBuffer);
            shader.SetBuffer(kernel_id, "CosIntDataResult", result.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);
            return result;
        }

        public FloatTensor AcosGPU(FloatTensor result)
        {
            int kernel_id = shader.FindKernel("AcosInt");

            shader.SetBuffer(kernel_id, "AcosIntData",this.DataBuffer);
            shader.SetBuffer(kernel_id, "AcosIntDataResult", result.DataBuffer);
            shader.Dispatch(kernel_id, this.size,1,1);
            return result;
        }

        public IntTensor AbsGPU(IntTensor result)
        {
            int kernel_id = shader.FindKernel("AbsElemInt");

            shader.SetBuffer(kernel_id, "AbsElemIntData", this.DataBuffer);
            shader.SetBuffer(kernel_id, "AbsElemIntDataResult", result.DataBuffer);

            shader.Dispatch(kernel_id, this.size, 1, 1);

            return result;
        }

        public void AbsGPU_()
        {
            int kernel_id = shader.FindKernel("AbsElemInt_");
            shader.SetBuffer(kernel_id, "AbsElemIntData_", this.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);
        }

        public IntTensor NegGPU(IntTensor result)
        {
            int kernel_id = shader.FindKernel("NegateInt");
            shader.SetBuffer(kernel_id, "NegateIntData", this.DataBuffer);
            shader.SetBuffer(kernel_id, "NegateIntResult", result.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);

            return result;
        }

        public void NegGPU_()
        {
            int kernel_id = shader.FindKernel("NegateInt_");
            shader.SetBuffer(kernel_id, "NegateIntData_", this.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);
        }

        public IntTensor SubGPU(IntTensor tensor, IntTensor result)
        {
            int kernel_id = shader.FindKernel("SubElemInt");

            shader.SetBuffer(kernel_id, "SubElemIntDataA", this.DataBuffer);
            shader.SetBuffer(kernel_id, "SubElemIntDataB", tensor.DataBuffer);
            shader.SetBuffer(kernel_id, "SubElemIntDataResult", result.DataBuffer);

            shader.Dispatch(kernel_id, this.size, 1, 1);

            return result;
        }

        public void SubGPU_(IntTensor tensor)
        {
            int kernel_id = shader.FindKernel("SubElemInt_");
            shader.SetBuffer(kernel_id, "SubElemIntDataA_", this.DataBuffer);
            shader.SetBuffer(kernel_id, "SubElemIntDataB_", tensor.DataBuffer);

            shader.Dispatch(kernel_id, this.size, 1, 1);
        }

        public IntTensor UnfoldGPU(int[] new_shape, int size, int dimSize, int sizeBeforeDim, int sizeAfterDim, int step)
        {
            IntTensor result = factory.Create(new_shape);
            result.Gpu(shader);

            int kernel_id = shader.FindKernel("UnfoldInt");

            var sizeBuffer = SendIntToGpu(kernel_id, size, "UnfoldIntSize");
            var dimSizeBuffer = SendIntToGpu(kernel_id, dimSize, "UnfoldIntDimSize");
            var sizeBeforeDimBuffer = SendIntToGpu(kernel_id, sizeBeforeDim, "UnfoldIntSizeBeforeDim");
            var sizeAfterDimBuffer = SendIntToGpu(kernel_id, sizeAfterDim, "UnfoldIntSizeAfterDim");
            var stepBuffer = SendIntToGpu(kernel_id, step, "UnfoldIntStep");
            shader.SetBuffer(kernel_id, "UnfoldIntData", this.DataBuffer);
            shader.SetBuffer(kernel_id, "UnfoldIntResult", result.DataBuffer);

            shader.Dispatch(kernel_id, result.size, 1, 1);

            sizeBuffer.Release();
            dimSizeBuffer.Release();
            sizeBeforeDimBuffer.Release();
            sizeAfterDimBuffer.Release();
            stepBuffer.Release();

            return result;
        }

        private ComputeBuffer SendFloatToGpu(int kernel, float value, string name)
        {
            float[] scalarArray = new float[1];
            scalarArray[0] = value;

            var scalarBuffer = new ComputeBuffer(1, sizeof(float));
            scalarBuffer.SetData(scalarArray);
            shader.SetBuffer(kernel, name, scalarBuffer);

            return scalarBuffer;
        }

        private ComputeBuffer SendIntToGpu(int kernel, int value, string name)
        {
            int[] array = new int[1];
            array[0] = value;

            var arrayBuffer = new ComputeBuffer(1, sizeof(float));
            arrayBuffer.SetData(array);
            shader.SetBuffer(kernel, name, arrayBuffer);

            return arrayBuffer;
        }

    }
}