using System.Diagnostics;
using System.Threading;
using System.Collections.Concurrent;
using NetMQ;
using NetMQ.Sockets;
using System;
using UnityEngine;

namespace OpenMined.Network.Servers
{

    public interface NetMqDelegate
    {
        void ProcessMessage(string json_message, MonoBehaviour owner, Action<string> response);
    }

    public class NetMqPublisher
    {
        private readonly Thread _listenerWorker;
        private bool _listenerCancelled;
        private NetMqDelegate _messageDelegate;
        private MonoBehaviour owner;
        private readonly Stopwatch _contactWatch;
        private const long ContactThreshold = 1000;
        public bool Connected;

        private readonly ConcurrentQueue<Request> _requestQueue = new ConcurrentQueue<Request>();

		public struct Request
		{
			public RouterSocket router;
			public string identity;
			public string message;

			public Request (RouterSocket _router, string _identity, string _message)
			{
				router = _router;
				identity = _identity;
				message = _message;
			}
		}

		private void ListenerWork ()
		{
			AsyncIO.ForceDotNet.Force ();

			using (var server = new RouterSocket ()) {
				server.Bind ("tcp://*:5555");

				while (!_listenerCancelled) {
					//server.SkipFrame(); // to skip identity

					string identity;
					if (!server.TryReceiveFrameString (out identity))
						continue;
					//UnityEngine.Debug.LogFormat ("identity {0}", identity);

					string message;
					if (!server.TryReceiveFrameString (out message))
						continue;
					//UnityEngine.Debug.LogFormat ("message {0}", message);

					//server.SendMoreFrame(identity).SendFrame("message");
					Request request = new Request (server, identity, message);
					_requestQueue.Enqueue (request);
				}
			}

			NetMQConfig.Cleanup ();
		}

        public NetMqPublisher (NetMqDelegate messageDelegate, MonoBehaviour owner)
		{
			_messageDelegate = messageDelegate;
			_contactWatch = new Stopwatch ();
			_contactWatch.Start ();
			_listenerWorker = new Thread (ListenerWork);
            this.owner = owner;
		}

		public void Start ()
		{
			_listenerCancelled = false;
			_listenerWorker.Start ();
		}

		public void Update ()
		{
			while (!_requestQueue.IsEmpty) {
				Request request;
				if (_requestQueue.TryDequeue (out request)) {
                    _messageDelegate.ProcessMessage(request.message, this.owner, (response) => {
                        request.router.SendMoreFrame(request.identity);
                        request.router.SendFrame(response);    
                    });
				} else {
					break;
				}
			}
		}

		public void Stop ()
		{
			_listenerCancelled = true;
			_listenerWorker.Join ();
		}
	}
}
