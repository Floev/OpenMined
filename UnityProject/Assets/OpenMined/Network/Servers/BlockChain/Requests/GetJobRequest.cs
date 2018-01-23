using UnityEngine;
using System.Collections;
using OpenMined.Network.Servers.BlockChain.Response;

namespace OpenMined.Network.Servers.BlockChain.Requests
{
    public class GetJobRequest : BaseRequest<GetJobResponse>
    {

        public string jobId;

        public GetJobRequest(string jobId) : base(Method.GET, "job")
        {
            this.jobId = jobId;
        }

        public override UnityEngine.Networking.UnityWebRequest GetWebRequest()
        {
            return GetRequest(subPath: "/" + this.jobId);
        }
    }
}