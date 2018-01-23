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
            controller = new SyftController(shader);

            _netMqPublisher = new NetMqPublisher(controller, this);
			_netMqPublisher.Start();
		}

		private void Update()
		{
			_netMqPublisher.Update();
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
