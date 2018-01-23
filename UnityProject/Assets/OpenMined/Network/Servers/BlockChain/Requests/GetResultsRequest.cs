using UnityEngine;
using System.Collections.Generic;
using OpenMined.Network.Servers.BlockChain.Response;

namespace OpenMined.Network.Servers.BlockChain.Requests
{
    public class GetResultsRequest: BaseRequest<GetResultsResponse>
    {
        public string jobAddress;

        public GetResultsRequest(string jobAddress) : base(Method.GET, "results")
        {
            this.jobAddress = jobAddress;
        }

        public override UnityEngine.Networking.UnityWebRequest GetWebRequest()
        {
            return GetRequest(subPath: "/" + this.jobAddress);
        }
    }
}
