# PSVita-LLM

> “Your scientists were so preoccupied with whether or not they could, they didn't stop to think if they should.”

After blowing the dust off my PS Vita to play *LittleBigPlanet*, a stray question popped up:

**Could an eleven-year-old handheld run an LLM?**

Turns out it can! **PSVita-LLM** runs a modified version of `llama2.c` to load and infer the TinyStories 260K and 15M checkpoints right on the Vita.


| Model | Parameters | File size | Inference speed (PCH-1000 Overclocked @ 555 MHz) |
|-------|------------|-----------|---------------------------------------------------|
| TinyStories-260K | 0.26 M | 1 MB | ≈ 120 tok/s  |
| TinyStories-15M | 15 M | 60 MB | ≈ 1.8 tok/s |



## Features

- **Interactive Model Selector:** On startup, the application scans the `ux0:data/` directory and presents a menu of all compatible models it finds.
- **Full "Game Loop":** After a story is generated, you can choose to generate another, return to the model selection screen to switch models, or exit the application cleanly.

## How to Use

1.  **Install the VPK:** Transfer the `psvita-llm.vpk` file to your Vita and install it using VitaShell.
2.  **Download Models & Tokenizers:** [Pending, I will provide a script for users to run]
4.  **Place Files:** Copy your `.bin` model files and their corresponding tokenizer files (`tokenizer.bin`, `tok512.bin`) to the `ux0:data/` directory on your PS Vita's memory card.
5.  **Launch the App:** 


## Building from Source

To build this project yourself, you will need a working [VitaSDK](https://vitasdk.org/) installation.

Once the SDK is configured, clone the repository and run:

```bash
cmake .
```



