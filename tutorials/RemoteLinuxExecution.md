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
