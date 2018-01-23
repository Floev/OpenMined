using UnityEngine;
using System.Collections;
using Newtonsoft.Json.Linq;

namespace OpenMined.Network.Servers.Ipfs
{
    public class IpfsJobConfig
    {
        [SerializeField] public float lr;
        [SerializeField] public string criterion;
        [SerializeField] public int iters;

        public IpfsJobConfig(float lr, string criterion, int iters)
        {
            this.lr = lr;
            this.criterion = criterion;
            this.iters = iters;
        }
    }
}
