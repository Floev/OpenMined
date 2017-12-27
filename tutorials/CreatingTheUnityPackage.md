## Creating the Unity Package

We export OpenMined to a Unity package so that it can be dropped into other projects. We started doing this for our integration with Unity's [ml-agents](https://github.com/Unity-Technologies/ml-agents).

### Export the Package

From the `Project` pane in Unity, right click the OpenMined dir and select `Export Package...`

Select only the following components:

* AsyncIO (dll)
* NetMQ (dll)
* Network (entire dir)
* Syft (entire dir)

Click `Export...` and save the package to `./OpenMined/dist/OpenMined.unitypackage`

More info on exporting: https://docs.unity3d.com/Manual/HOWTO-exportpackage.html

### Import the Package

From another Unity project (not OpenMined), right click within the `Project` pane and select `Import Package > Custom Package...`.

Select the OpenMined package from `./OpenMined/dist/OpenMined.unitypackage`.

OpenMined should now be successfully imported.

More info on importing: https://docs.unity3d.com/Manual/AssetPackages.html

