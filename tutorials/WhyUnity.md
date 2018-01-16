# Why Unity?

Are you wondering why is **OpenMined using Unity to build a Deep Learning library** that enables Federated Learning via Homomorphic Encryption and Blockchain technology? Hey! In fact that's a great question, and you are in the right place.

## GPU everywhere for everyone

The big one reason is that we want **very versatile GPU support**, because **we also want it to be the most versatile Deep Learning framework on the market**. Every other deep learning framework basically jumps on the CUDA bandwagon. That sounds good because Nvidia GPUs with CUDA works like a charm but in practice is not that good. **Not everyone has a device with a CUDA compatible graphic card**, right?

Not a problem for us, we're targeting **deep learning on devices that every-day people own** (with minimal code replication). As it turns out, **Unity is absolutely perfect** for this

First and foremost, because the **most powerful GPU** that **everyday consumers own** is a **Playstation** or **Xbox**. For example, the latest Xbox has a GPU that is [comparable](https://www.techpowerup.com/gpudb/2977/xbox-one-x-gpu) with a top of the line NVIDIA deep learning card from a couple years ago.

## Unleash world's GPUs

When the thing really gets cool is when you realize that **Unity has the ability to cross compile code to** almost **every GPU on the planet**, as it sounds. 

By packaging PySyft's backend in the Unity game engine we can cross-compile to:

- Android
- iOS
- PlayStation
- Xbox
- Linux (using the GPU on the laptop/desktop screen)
- Mac (using the GPU on the laptop/desktop screen)
- Windows (using the GPU on the laptop/desktop screen)
- Smart TVs (using the GPU in the television)
- etc..... 

And all of that also **providing private access to amazing datasets**

Not convinced yet? Let's take a closer look

### Smartphones 
Nowadays smartphones are getting faster and more powerful. Maybe thought for gaming, but they're having more and more powerful GPUs.

#### Android
Most high-end devices carry Snapdragon 835 and similar processors. Look at that beautiful Adreno 540 GPU waiting on your smartphone to do incredible things. 
![snapdragon_835](https://github.com/OpenMined/OpenMined/blob/master/images/WhyUnity/snapdragon_835.png)

Or even better! Smartphones has become the first devices that have **neural dedicated processors**. The new Kirin 970 mounted in the Huawei Mate 10 is an example of that.
![kirin_970](https://github.com/OpenMined/OpenMined/blob/master/images/WhyUnity/kirin_970.PNG)

#### iOS

Apple's **A11 Bionic** processor includes **dedicated neural network hardware** that Apple calls a *"Neural Engine"*. This neural network hardware can perform up to 600 billion operations per second. The neural engine allows Apple to implement neural network and machine learning in a more energy-efficient manner than using either the main CPU or the GPU.
![apple_a11](https://github.com/OpenMined/OpenMined/blob/master/images/WhyUnity/apple_a11.jpg)

### PlayStation & Xbox

Playstation / Xbox game consoles have super powerful GPUs, 1TB local storage, wired internet connection, and sit idle 95% of the time. Nothing more to say.

### Desktop / Laptop computers

Open the door to Nvidia GPUs without CUDA, AMDs GPUs, Intel GPUs, all the chipset integrated GPUs, etc and in every OS.

### Future

Probably almost everything else to come...


