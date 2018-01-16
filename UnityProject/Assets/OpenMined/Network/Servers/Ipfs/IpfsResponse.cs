using UnityEngine;
using System.Collections;
using System;

namespace OpenMined.Network.Servers
{
    [Serializable]
    public class IpfsResponse
    {
        public string Name;
        public string Hash;
        public string Size;
    }    
}