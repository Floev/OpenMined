using UnityEngine;
using System.Collections;
using OpenMined.Network.Servers.BlockChain.Response;

namespace OpenMined.Network.Servers.BlockChain.Requests
{
    public class GetAvailableJobIdRequest: BaseRequest<GetAvailableJobIdResponse>
    {
        public GetAvailableJobIdRequest() : base(Method.GET, "availableJobId")
        {
        }

        public override UnityEngine.Networking.UnityWebRequest GetWebRequest()
        {
            return GetRequest();
        }
    }
}