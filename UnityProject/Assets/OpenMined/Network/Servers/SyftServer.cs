using UnityEngine;
using OpenMined.Network.Controllers;
using System.Collections;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace OpenMined.Network.Servers
{

	public class SyftServer : MonoBehaviour
	{
		public bool Connected;
		private NetMqPublisher _netMqPublisher;
		private string _response;

		public SyftController controller;

		[SerializeField] private ComputeShader shader;

		private void Start()
		{
			_netMqPublisher = new NetMqPublisher(HandleMessage);
			_netMqPublisher.Start();

			controller = new SyftController(shader);
            var request = new Request();

            //var ipfsAddress = request.modelResponse.configAddress;

            //IpfsModel model = Ipfs.GetModel(ipfsAddress);
            //if (model != null)
            //{
            //    Debug.Log("Got the IpfsModel: " + model.input);
                
            //    var g = new Controllers.Grid(controller);
            //    //g.TrainModel(model);
            //}
		}

		private void Update()
		{
			_netMqPublisher.Update();
		}

		private string HandleMessage(string message)
		{
			//Debug.LogFormat("HandleMessage... {0}", message);
			return controller.processMessage(message, this);
		}

		private void OnDestroy()
		{
			_netMqPublisher.Stop();
		}

		public ComputeShader Shader {
			get { return shader; }
		}

	}
}
