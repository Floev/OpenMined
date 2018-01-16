using UnityEngine;
using System.Collections;
using Newtonsoft.Json.Linq;
using System.IO;
using Newtonsoft.Json;
using OpenMined.Network.Controllers;

namespace OpenMined.Network.Servers
{
    public class BlockChain : MonoBehaviour
    {
        private Coroutine routine;
 
        public void Start()
        {
            var o = ReadConfig();

            if (o["trainer"].ToObject<bool>())
            {
                Debug.Log("POLLING");
                PollNext();
            }
        }

        void PollNext()
        {
            this.routine = StartCoroutine(PollNetwork());
        }

        IEnumerator PollNetwork()
        {
            var request = new Request();

            yield return request.GetBlockNumber(this);
            yield return request.GetModel(this);
            
            var ipfsAddress = request.modelResponse.configAddress;

            Debug.Log("IPFS address: " + ipfsAddress);

            IpfsModel model = Ipfs.GetModel(ipfsAddress);
            if (model != null)
            {
                Debug.Log("Got the IpfsModel: " + model.input);
                
                //var g = new Controllers.Grid(controller);
                //g.TrainModel(model);
            }

            Debug.Log("Blockchain polled");

            yield return new WaitForSeconds(10);
            PollNext();
        }
        
        JObject ReadConfig()
        {
            using (StreamReader reader = File.OpenText("Assets/OpenMined/Config/config.json"))
            {
                return (JObject)JToken.ReadFrom(new JsonTextReader(reader));
            }
        }
    }
}