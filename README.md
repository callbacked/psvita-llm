# PSVita-LLM

> “Your scientists were so preoccupied with whether or not they could, they didn't stop to think if they should.”

After blowing the dust off my PS Vita to play *LittleBigPlanet*, a stray question popped up:

**Could it run an LLM?**

Turns out it can! **PSVita-LLM** runs a modified version of `llama2.c` to load and infer the TinyStories 260K and 15M checkpoints right on the Vita.


| Model | Parameters | File size | Inference speed (PCH-1000 Overclocked @ 555 MHz) | Results |
|-------|------------|-----------|---------------------------------------------------|---------|
| TinyStories-260K | 0.26 M | 1 MB | ≈ 120 tok/s | <img src="https://github.com/user-attachments/assets/9c643fa8-0ee4-44d6-958d-914520dbc3da" width="200"> |
| TinyStories-15M | 15 M | 60 MB | ≈ 1.8 tok/s | <img src="https://github.com/user-attachments/assets/b5be21ad-2827-448d-86cf-54528be79bb7" width="200"> |

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



