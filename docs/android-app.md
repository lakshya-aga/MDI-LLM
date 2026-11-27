# Android app (native secondary node)

The app in [`android/`](../android) turns a phone into a secondary node with no Termux,
Python or PyTorch. It is a thin Kotlin shell around the C++ node in
[`native/`](../native):

- a **foreground service** keeps the node running with the screen off. It holds a
  partial wakelock and a low-latency Wi-Fi lock;
- **one screen** to import the nodes config, start and stop the node, and watch its
  status (layers, memory, per-pass latency) and log.

The laptop still runs the Python starter. The protocol is in
[wire-format.md](wire-format.md). Python and native secondaries can be mixed in one
ring.

## Why native

| | Termux + PyTorch | App (C++ node) |
|---|---|---|
| Runtime overhead | ~200–300 MB (Python + torch) | a few MB (a small `libmdinode.so`, no interpreter) |
| Weights in RAM | float32 by default (bf16 only with fast CPU kernels) | kept as sent: **bf16 for TinyLlama, half of float32**, computed with NEON |
| Setup per phone | Termux, `pkg install python-torch`, git clone | install one APK |

## Build

Requirements: JDK 17 and the Android SDK (platform 35). Gradle installs the NDK and
build-tools it needs on the first build.

- **Android Studio:** *Open* the `android/` folder, then *Build → Build APK(s)*.
- **Command line** (macOS with Homebrew; other systems are similar):

  ```bash
  brew install openjdk@17 && brew install --cask android-commandlinetools
  export JAVA_HOME=$(brew --prefix openjdk@17) ANDROID_HOME=$(brew --prefix)/share/android-commandlinetools
  yes | sdkmanager --licenses && sdkmanager "platforms;android-35" "platform-tools" "cmake;3.22.1"
  cd android && echo "sdk.dir=$ANDROID_HOME" > local.properties && ./gradlew assembleRelease
  ```

  The APK (about 3 MB) is `android/app/build/outputs/apk/release/app-release.apk`. It is
  signed with the debug key so it can be sideloaded: `adb install -r app-release.apk`,
  or copy it to the phone and open it.

The APK contains `arm64-v8a` (phones, NEON kernels) and `x86_64` (emulator) libraries.

### Try it in the emulator first

With an arm64 emulator running (`sdkmanager "emulator"
"system-images;android-35;google_apis;arm64-v8a"`, then `avdmanager create avd`), use
adb port forwarding so the whole ring runs on `127.0.0.1`:

```bash
adb install -r android/app/build/outputs/apk/release/app-release.apk
adb forward tcp:8089 tcp:8089 && adb forward tcp:5090 tcp:5090   # starter -> node
adb reverse tcp:5088 tcp:5088                                     # node -> starter
```

In the app, set the IP to `127.0.0.1`, keep the default ports, and tap **Start node**.
Then run the starter with a nodes config that has one secondary at `127.0.0.1`
(ports 8089/5090/5091). Run `adb forward --remove-all` afterwards; until then adb holds
those ports on the laptop.

## Use

1. On the laptop, prepare the model and write the nodes config, as in
   [setup-android.md](setup-android.md) steps 2–3. Each phone's `addr` must be its
   Wi-Fi IP, because nodes accept ring connections only from the configured address.
2. Copy the config JSON to each phone, open the app and tap **Import nodes config…**.
   The app selects the secondary index whose `addr` matches the phone's IP. You can
   also enter the index and tap **Apply index**.
3. Optional: **Battery optimization settings** → set the app to *Unrestricted* (or
   *Don't optimize*).
4. Tap **Start node** on every phone, then start the laptop:

   ```bash
   cd src && python starter.py --ckpt checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --nodes-config settings_distr/phones.json --n-samples 3 --n-tokens 200 \
     --sequence-length 512 --prompt "Who are you?" -v
   ```

5. After a run, the node goes back to waiting, so you can start the starter again
   without touching the phones. **Stop node** (or *Stop* in the notification) shuts it
   down.

### Local weights (optional)

Sending the weights over Wi-Fi takes a while for big chunks. To skip the transfer:

```bash
cd src && python scripts/export_chunks.py checkpoints/<model>/chunks/3nodes
adb push checkpoints/<model>/chunks/3nodes/model_secondary0.safetensors /sdcard/Download/
```

In the app, tap **Import chunk…**, select the file, and tick **Use the local chunk**.
The node memory-maps the file, so it isn't copied into RAM up front. Add
`--dtype bfloat16` to `export_chunks.py` to halve the file if it is float32.

### Settings

- **Threads:** 0 uses all cores. On phones with a few big cores, setting it to the
  number of big cores can be faster. Compare the *avg ms per pass* figure.
- **Weights stored as:** `auto` keeps the dtype the starter sends. `float16` or
  `bfloat16` halves the memory of float32 chunks. `float32` uses full precision and
  twice the memory.

## Command-line node (no app)

The same node runs on Linux, macOS, or Android through `adb shell` / Termux, as a
drop-in replacement for `secondary.py`:

```bash
cd native && cmake -B build && cmake --build build -j
./build/mdi_node --nodes-config ../src/settings_distr/phones.json 0 [--chunk model_secondary0.safetensors] -v
```

## Troubleshooting

- **The starter retries "Unable to reach node" forever:** the phone's IP doesn't match
  the config, or the network blocks device-to-device traffic (guest or public Wi-Fi). A
  phone hotspot works.
- **"Rejected connection from …" in the log:** the previous node's `addr` in the config
  isn't the address it actually connects from. Fix that node's entry.
- **Initialization fails with an out-of-memory error, or the app is killed:** use more
  nodes, a shorter `--sequence-length`, or bf16 weights.
