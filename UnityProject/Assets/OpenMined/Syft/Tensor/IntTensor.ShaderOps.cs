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

    }
}