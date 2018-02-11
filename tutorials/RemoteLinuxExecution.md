
# Executing OpenMined on a Remote Linux Server.

You've fully installed Unity, PySyft, and OpenMined on your Linux (e.g., Ubuntu 16.04) server, and you're logged in over ssh.


0. Your Linux machine has an X (X11) server running.  If it doesn't, you need to install one.  But we're not going to use X11 forwarding, i.e. no "ssh -Y" sort of connection; we will instead run Unity on the machine's local monitor (even though you won't be able to see it).  If your server *has no* monitor, then [follow this guide to set up a 'fake' X display](https://towardsdatascience.com/how-to-run-unity-on-amazon-cloud-or-without-monitor-3c10ce022639).  *Note that you need permission to use this X display, i.e. if you log in to a computer that someone else is running the X session on, you will be locked out, unless they grant you permission via [xhost +](https://www.x.org/archive/X11R6.8.1/doc/xhost.1.html).


1. Create a new directory UnityProject/Assets/Editor/ and inside place a C# script containing a method (MyEditorScript.Start) that will push the "Play" button for you.


        $ cat Assets/Editor/MyEditorScript.cs
        using UnityEngine;
        using System.Collections;
        using UnityEditor;

        class MyEditorScript { 
            static void Start() { 
                  UnityEditor.EditorApplication.ExecuteMenuItem("Edit/Play"); 
             }
        }

[credit: https://answers.unity.com/questions/1136444/automating-playmode-from-batch-and-c-executemethod.html]


2. Unity needs a MyEditorScript.cs.meta file as well, which in theory you can only get by running the Unity GUI.   **OR** you can just copy over a *different* .meta file, and manually make up a new "guid" string which is probably a bad idea but it works: (I changed a leading '7' to a '6' LOL)...

        $ cat Assets/Editor/MyEditorScript.cs.meta 
        fileFormatVersion: 2
        guid: 61d2145c185b043f8956955bd1c464b3
        timeCreated: 1512580258
        licenseType: Free
        MonoImporter:
          externalObjects: {}
          serializedVersion: 2
          defaultReferences: []
          executionOrder: 0
          icon: {instanceID: 0}
          userData: 
          assetBundleName: 
          assetBundleVariant: 

Same with the Editor/ directory: it needs an Editor.meta file which you can get by copying a different directory's .meta file and modifying the guid.  Probably better to actually let the GUI assign guids, but if you literally have no way to get there, this works. 

3. Invoke Unity from the command line, *using the main console's display* for the project you want, and execute the Play button script:

        $ DISPLAY=:0 ~/Unity-2017.3.0b1/Editor/Unity -projectPath ~/OpenMined/UnityProject -executeMethod MyEditorScript.Start

5. That's it!  You can now try the various OpenMined demos & tutorials remotely, e.g. by [port-forwarding your jupyter notebooks](https://drscotthawley.github.io/How-To-Port-Forward-Jupyter-Notebooks/). 

## TODO: Viewing status
You won't be able to see the OpenMined log messages that appear in the Unity window.  
 OpenMined authors: Presumably there's a  find a way to access those logs from the command line?
=======
# Preparing OpenMined for Remote Linux Execution

*Disclaimer: The following is a work in progresss. We do **not** yet have a successful build.  Feel free to help!*

You've installed Unity, PySyft, and OpenMined on your Linux (e.g., Ubuntu 16.04) server, and you're logged in over ssh via "ssh -X".

Trying to run the Unity Editor will give you "Failed to initialize graphics" fatal error.

What you need to do is build the OpenMined Unity app for Linux, using the "headless" build mode of the Unity Editor.

But there's a problem: the OpenMined Unity project has a file "Newtonsoft.Json.dll" which references a bunch of "System*.dll" files that Unity can't find, even though they *are* installed.

To ensure that the Unity builder can find them, execute the following:

    cd OpenMined/UnityProject/Assets 
    ln -s {YOUR_UNITY_PATH}/Editor/Data/MonoBleedingEdge/lib/mono/4.5/Facades/System.*dll .

Now you can build using the following command:

    {YOUR_UNITY_PATH}/Unity -batchmode -nographics -projectPath {YOUR_OPENMINED_PATH}/UnityProject -logFile mylog  -buildLinuxUniversalPlayer OpenMinedApp -enableIncompatibleAssetDowngrade -quit

You'll see the following errors appear first...

    ALSA lib confmisc.c:768:(parse_card) cannot find card '0'
    ALSA lib conf.c:4292:(_snd_config_evaluate) function snd_func_card_driver returned error: No such file or directory
    ALSA lib confmisc.c:392:(snd_func_concat) error evaluating strings
    ALSA lib conf.c:4292:(_snd_config_evaluate) function snd_func_concat returned error: No such file or directory
    ALSA lib confmisc.c:1251:(snd_func_refer) error evaluating name
    ALSA lib conf.c:4292:(_snd_config_evaluate) function snd_func_refer returned error: No such file or directory
    ALSA lib conf.c:4771:(snd_config_expand) Evaluate error: No such file or directory
    ALSA lib pcm.c:2266:(snd_pcm_open_noupdate) Unknown PCM default
    /home/builduser/buildslave/unity/build/Editor/Platform/Linux/UsbDevices.cpp:UsbDevicesQuery
    [0113/145735:ERROR:browser_main_loop.cc(161)] Running without the SUID sandbox! See https://code.google.com/p/chromium/wiki/LinuxSUIDSandboxDevelopment for more information on developing with the sandbox on.
    [0113/145736:ERROR:gl_context_glx.cc(68)] Failed to create GL context with glXCreateNewContext.
    [0113/145736:ERROR:gpu_info_collector.cc(41)] gfx::GLContext::CreateGLContext failed
    [0113/145736:ERROR:gpu_info_collector.cc(95)] Could not create context for info collection.
    [0113/145736:ERROR:gpu_main.cc(402)] gpu::CollectGraphicsInfo failed (fatal).
    [0113/145737:ERROR:gpu_child_thread.cc(143)] Exiting GPU process due to errors during initialization

...and then the build will proceed (or "linger around") for a *really* long time.  It takes so long because after Unity decides it can't use GPU acceleration to build, the CPU usage drops to around 7%.   

    $ top
    PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND                                                                                                                       
    7714 shawley   20   0 2614172 616368 128896 R   9.0  0.9   1:31.86 Unity                                                                                                                         


So estimate however long you think it should take to do the build, and multiply by a factor of ~16.

...and eventually it will exit with the message

    debugger-agent: Unable to listen on 27
    
    
That [debugger-agent message is a red herring](https://forum.unity.com/threads/6572-debugger-agent-unable-to-listen-on-27.500387/).  Your build failed.  Looking at the mylog file [mirrored here](http://hedges.belmont.edu/~shawley/latest_unity_build_log.txt), one finds the following error:

    System.Windows.Forms.dll assembly is referenced by user code, but is not supported on StandaloneLinuxUniversal platform. Various failures might follow.

So, it's not surprising that "System.Windows.Forms.dll" is "not supported on StandaloneLinuxUniversal platform" -- what I'm confused about is why it's even *trying* to link a Windows file for Linux build.   

Posting this question to Unity3D forums: [Link Here](https://answers.unity.com/questions/1454241/systemwindowsformsdll-assembly-is-referenced-by-us.html)

