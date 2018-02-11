using System;

namespace OpenMined.Network.Utils
{
    [Serializable]
    public class GridConfiguration 
    {
        public int model;
        public float lr;
        public string criterion;
        public int iters;
    }    
}