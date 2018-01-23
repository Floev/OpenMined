using UnityEngine;
using System.Collections.Generic;
using OpenMined.Network.Servers.BlockChain.Response;
using Newtonsoft.Json;
using UnityEngine.Networking;

namespace OpenMined.Network.Servers.BlockChain.Requests
{
    public class AddResultRequest: BaseRequest<GenericResponse>
    {
        public string jobAddress;
        public string resultAddress;

        public AddResultRequest(string jobAddress, string resultAddress): base(Method.POST, "result")
        {
            this.jobAddress = jobAddress;
            this.resultAddress = resultAddress;
        }

        public override UnityEngine.Networking.UnityWebRequest GetWebRequest()
        {
            WWWForm data = new WWWForm();
            data.AddField("jobAddress", this.jobAddress);
            data.AddField("resultAddress", this.resultAddress);

            return base.GetRequest(
                postBody: data
            );
        }
    }
}
