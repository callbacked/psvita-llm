# PSVita-LLM

> “Your scientists were so preoccupied with whether or not they could, they didn't stop to think if they should.”

After blowing the dust off my PS Vita to play *LittleBigPlanet*, a stray question popped up:

**Could it run an LLM?**

Turns out it can! **PSVita-LLM** runs a modified version of `llama2.c` to load and infer the TinyStories 260K and 15M checkpoints right on the Vita.


| Model | Parameters | File size | Inference speed (PCH-1000 Overclocked @ 555 MHz) | Results |
|-------|------------|-----------|---------------------------------------------------|---------|
| TinyStories-260K | 0.26 M | 1 MB | ≈ 120 tok/s | <img src="https://github.com/user-attachments/assets/1f1924cb-6be6-4b6e-8bee-5eef46ca1782" width="200"> |
| TinyStories-15M | 15 M | 60 MB | ≈ 1.8 tok/s | <img src="https://github.com/user-attachments/assets/0b6ab797-211d-47fb-afcb-13d788cf44a8" width="200"> |


## Features

- **Interactive Model Selector:** On startup, the app will prompt the user to a download model to start should it detect that there are no models downloaded.
- **Full "Game Loop":** After a story is generated, you can choose to generate another, return to the model selection screen to switch/manage models, or exit the app completely.

## How to Use

1.  **Install the VPK:** Transfer the `psvita-llm.vpk` file to your Vita and install it using VitaShell.
2.  **Download Models & Tokenizers:** Upon first boot, the program will give you the models available to download. You can delete models and download any other model after that in the "Manage local models.." menu.
3. **Enjoy!**


## Building from Source

To build this project yourself, you will need a working [VitaSDK](https://vitasdk.org/) installation.

Once the SDK is configured, clone the repository and run:

```bash
cmake .
```

This project would not be possible without Andrej Karpathy's foundational work on `llama2.c`. What was done was more of a port job if anything, but a great way to learn!

### Future Improvements

Some notes that I want to leave in before I forget

*   **Breaking up some code:** Having everything in a single file like the original llama2.c file is pretty cool, but I should have seperated the networking code as I feel like that could be used in other projects that involve downloading stuff/doing curl calls on the internet, it'd be a good reference to have.

*   **Multithreading:** The current code has commented out `#pragma omp` directives. It's because OpenMP does not play nicely with the Vita's CPU. Leaving it on led to crashes upon generation. A significant performance boost could probably be seen by implementing a native multithreading solution using stuff in `SceThreadMgr` library in the sdk (?), especially for parallelizing the `matmul` ops in the transformer's forward pass. **For now this is all single threaded**.

But I'm out of my depth in terms of development with the SDK, but it is something worth considering should I give this project another look though.




