using UnityEngine;
using System.Collections;
using Newtonsoft.Json.Linq;

namespace OpenMined.Network.Servers.Ipfs
{
    public class IpfsJob
    {
        public string input;
        public string target;
        public JToken Model;
        public IpfsJobConfig config;

        public IpfsJob(string input, string target, JToken model, IpfsJobConfig config)
        {
            this.input = input;
            this.target = target;
            this.Model = model;
            this.config = config;
        }
    }
}
