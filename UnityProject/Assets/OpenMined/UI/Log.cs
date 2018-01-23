using UnityEngine;
using UnityEngine.UI;
using System;

namespace OpenMined.UI
{
    public class Log : MonoBehaviour
    {
        public Text logText;

        public static Log Instance;

        void Awake()
        {
            Instance = this;
        }

        void Start()
        {
            this.ClearText();
            this.SetText("<color=white>Started logging...</color>");
        }

        void OnEnable()
        {
            Application.logMessageReceived += HandleLog;
        }

        void OnDisable()
        {
            Application.logMessageReceived -= HandleLog;
        }

        void HandleLog(string logString, string stackTrace, LogType type)
        {
            this.SetText(logString);
        }

        public void ClearText()
        {
            //logText.text = "";
        }

        public void SetText(string text)
        {
            //UnityEngine.Debug.LogFormat(text);
            //logText.text += text + "\n";
        }

        public static void LogFormat(string format, params object[] args)
        {
            OpenMined.UI.Log.Instance.SetText(String.Format(format, args));
        }
    }
}
