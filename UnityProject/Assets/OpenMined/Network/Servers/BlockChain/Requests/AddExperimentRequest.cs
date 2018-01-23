using UnityEngine;
using System.Collections.Generic;
using UnityEngine.Networking;
using Newtonsoft.Json;
using OpenMined.Network.Servers.BlockChain.Response;


namespace OpenMined.Network.Servers.BlockChain.Requests
{
    public class AddExperimentRequest: BaseRequest<GenericResponse>
    {
        public string experimentAddress;
        public string[] jobAddresses;

        public AddExperimentRequest (string experimentAddress, string[] jobAddresses) : base (Method.POST, "experiment")
        {
            this.experimentAddress = experimentAddress;
            this.jobAddresses = jobAddresses;
        }

        override public UnityWebRequest GetWebRequest ()
        {
            WWWForm data = new WWWForm();
            data.AddField("experimentAddress", this.experimentAddress);
            data.AddField("jobAddresses", JsonConvert.SerializeObject(this.jobAddresses));

            return base.GetRequest(
                postBody: data
            );
        }
    }
}