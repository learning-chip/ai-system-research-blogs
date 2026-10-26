# Set up Cursor & Claude Code agent inside Ascend-Docker container on remote NPU server

Chinese version: [README_zh.md](README_zh.md)

Full code, Dockerfile, and devcontainer config: https://github.com/learning-chip/agent_docker_npu/tree/0.1.0

Goal of this guide:
- Give the agent access to NPU devices for autonomous "edit-compile-run" cycles
- Isolate agent execution within Docker to prevent accidental system-wide changes
- Package heavy CANN dependencies and various frameworks (PyTorch, PTO, TileLang, etc.) inside the docker image
- Minimal steps to get started!

## Outline

- [Prerequisite: Ascend-Docker engine and CANN image](#prerequisite-ascend-docker-engine-and-cann-image)
  - [Install Docker engine with Ascend runtime](#install-docker-engine-with-ascend-runtime)
  - [Build docker image with commonly-used NPU dependencies](#build-docker-image-with-commonly-used-npu-dependencies)
  - [Test docker container launch](#test-docker-container-launch)
  - [Prepare Dev container configuration](#prepare-dev-container-configuration)
- [Cursor agent in remote docker](#cursor-agent-in-remote-docker)
  - [Step1: Cursor plug-ins](#step1-cursor-plug-ins)
  - [Step2: Remote container connection](#step2-remote-container-connection)
  - [Step3: Execute commands in remote container](#step3-execute-commands-in-remote-container)
- [Claude Code in remote docker](#claude-code-in-remote-docker)
  - [Step1: VS Code plug-ins](#step1-vs-code-plug-ins)
  - [Step2: Remote container connection](#step2-remote-container-connection-1)
  - [Step3: Enable Claude Code extension](#step3-enable-claude-code-extension)


## Prerequisite: Ascend-Docker engine and CANN image

### Install Docker engine with Ascend runtime

On the remote server, first install standard Docker following the [official steps](https://docs.docker.com/engine/install/ubuntu/).

Then, install [Ascend Docker Runtime](https://gitcode.com/Ascend/mind-cluster/tree/master/component/ascend-docker-runtime) on top of standard Docker. Use `Ascend-docker-runtime*.run` files from the [release page](https://gitcode.com/Ascend/mind-cluster/releases).

Run `sudo usermod -aG docker $USER` so you do not need `sudo` for later `docker run` commands.

### Build docker image with commonly-used NPU dependencies

For the base image, use the officially maintained [cann-container-image Dockerfiles](https://github.com/Ascend/cann-container-image). Pre-built images are in the [quay.io registry](https://quay.io/repository/ascend/cann?tab=tags).

See the full file in [.devcontainer/Dockerfile](https://github.com/learning-chip/agent_docker_npu/blob/0.1.0/.devcontainer/Dockerfile). Important lines are:

```dockerfile
FROM quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11

RUN apt-get update && apt-get install -y \
    wget curl git vim \
    nodejs npm
# NOTE: nodejs and npm are only needed for claude code CLI

# for on-device execution
RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torch-npu==2.9.0 \
    && pip install --no-cache-dir numpy pyyaml

# extra utilities
RUN pip install --no-cache-dir \
    pytest pybind11 nanobind setuptools wheel \
    ipython jupyterlab matplotlib pandas

RUN npm install -g @anthropic-ai/claude-code
# NOTE: only for claude-code CLI; no need to install this if using remote VS Code extensions

# install other frameworks you need below
# ...
```

Build with `docker build -t agent_npu_cann:8.5.0 .`

### Test docker container launch

Remarks on the launch command below:
- `$HOST_MOUNT_DIR` is mounted from host to container, for all files & code repos that the agent can **read and edit**
- Set `:ro` (read-only) for host-side NPU dependencies such as the CANN driver, so the agent can **only read but not modify** them
- To restrict NPU access, you can pass fewer `--device=/dev/davinci*`
- `ANTHROPIC_API_KEY` is only needed for the Claude CLI, not for the Claude VS Code extension.

```bash
HOST_MOUNT_DIR=$HOME/work_code/workdir_for_agent  # agent can only access this directory

docker run --rm -it --ipc=host --privileged \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v $HOST_MOUNT_DIR:/workdir \
    -w /workdir \
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    agent_npu_cann:8.5.0 /bin/bash
```

### Prepare Dev container configuration

Both Cursor and VS Code will need a [Dev Container json](https://containers.dev/implementors/json_reference/) to connect the editor to the container runtime.

Put this `.devcontainer/devcontainer.json` file on your remote server:

```json
{
  "name": "agent_npu",
  "image": "agent_npu_cann:8.5.0",
  "workspaceFolder": "/workdir",
  "runArgs": [
    "--ipc=host",
    "--privileged",
    "--device=/dev/davinci0",
    "--device=/dev/davinci1",
    "--device=/dev/davinci2",
    "--device=/dev/davinci3",
    "--device=/dev/davinci4",
    "--device=/dev/davinci5",
    "--device=/dev/davinci6",
    "--device=/dev/davinci7",
    "--device=/dev/davinci_manager",
    "--device=/dev/devmm_svm",
    "--device=/dev/hisi_hdc"
  ],
  "mounts": [
    "type=bind,source=/usr/local/bin/npu-smi,target=/usr/local/bin/npu-smi,readonly",
    "type=bind,source=/usr/local/Ascend/driver,target=/usr/local/Ascend/driver,readonly",
    "type=bind,source=/etc/ascend_install.info,target=/etc/ascend_install.info,readonly",
    "type=bind,source=${localEnv:HOME}/work_code/workdir_for_agent,target=/workdir"
  ],
  "containerEnv": {
    "ANTHROPIC_API_KEY": "${localEnv:ANTHROPIC_API_KEY}"
  }
}
```

It mirrors the `docker run` command in the previous section. Change the `workdir_for_agent` in `"mounts"` to mount a different host directory. Change `"containerEnv"` to pass more environment variables. Change `"runArgs"` to mount fewer or more devices.

You can just clone this repository that contains both `Dockerfile` and `devcontainer.json`.

## Cursor agent in remote docker

### Step1: Cursor plug-ins

Install the Anysphere **Remote SSH** extension:

<img src="./fig/cursor-remote-ssh-extension-marketplace.png" alt="Cursor marketplace: Remote SSH extension" width="100%" />

Install the Anysphere **Dev Containers** extension:

<img src="./fig/cursor-dev-containers-extension-marketplace.png" alt="Cursor marketplace: Dev Containers extension" width="100%" />

## Step2: Remote container connection

Search for "Remote SSH" in the Command Palette (same shortcut as in VS Code: `Ctrl + Shift + P`, or `Command + Shift + P` on Mac).

<img src="./fig/cursor-remote-ssh-command-palette.png" alt="Cursor Command Palette: connect with Remote SSH" width="50%" />

When you open the folder that contains `.devcontainer/devcontainer.json`, a prompt should appear to reopen in the container.

<img src="./fig/cursor-reopen-in-container-notification.png" alt="Cursor: reopen folder in dev container prompt" width="50%" />

Otherwise, search for **"Dev Containers: Reopen in Container"** in the Command Palette (`Ctrl + Shift + P` / `Command + Shift + P`). Choose "Rebuild container" if you changed the Docker image.

<img src="./fig/cursor-dev-containers-reopen-command.png" alt="Cursor Command Palette: Dev Containers Reopen in Container" width="70%" />

Setting up the Docker instance can take tens of seconds the first time.

## Step3: Execute commands in remote container

Then, terminal commands and the **agent's commands** will run inside this remote NPU Docker environment.

If a bash terminal does not open, search for **"Create New Terminal (With Profile)"**.

<img src="./fig/cursor-new-terminal-with-profile.png" alt="Cursor Command Palette: Create New Terminal With Profile" width="50%" />

Verify that torch-npu runs in Cursor's integrated terminal:

<img src="./fig/cursor-terminal-pytorch-npu-smoke-test.png" alt="Cursor integrated terminal: PyTorch NPU device check" width="70%" />

<img src="./fig/cursor-terminal-npu-smi-output.png" alt="Cursor integrated terminal: npu-smi output" width="70%" />

When the agent asks to run commands, set it to "Run Everything" so it can compile and run NPU kernels and get closed-loop feedback.

<img src="./fig/cursor-agent-run-everything-notification.png" alt="Cursor agent: Run Everything permission prompt" width="30%" />

Or change it in the settings UI (avoid this permissive option when you are not inside a container):

<img src="./fig/cursor-agent-terminal-run-mode-setting.png" alt="Cursor settings: Agent terminal run mode" width="70%" />

A small but annoying preference: I type long prompts, so I prefer "Enter" rather than "Shift + Enter" for line breaks:

<img src="./fig/cursor-chat-enter-for-line-break-setting.png" alt="Cursor settings: Enter for line break in chat" width="70%" />


## Claude Code in remote docker

There are many ways to use Claude Code; this section only describes the "Cursor-like" workflow using the [VS Code Claude extension](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code).

### Step1: VS Code plug-ins

In VS Code on your local computer, install the official [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) and [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extensions.

## Step2: Remote container connection

Connect to the remote server ("Remote SSH: Connect to Host" in the Command Palette).

<img src="./fig/vscode-remote-ssh-host-connected.png" alt="VS Code: Remote SSH connected to host" width="50%" />

In the remote VS Code window, open a directory that contains a `.devcontainer/` subdirectory (with a `devcontainer.json` file inside). You can clone this repo and open it in VS Code; VS Code will then offer **Reopen in Container**.

<img src="./fig/vscode-reopen-in-container-notification.png" alt="VS Code: reopen folder in dev container prompt" width="50%" />

Otherwise, search for **"Dev Containers: Reopen in Container"** in the Command Palette. Choose "Rebuild container" if you changed the Docker image.

<img src="./fig/vscode-dev-containers-reopen-command.png" alt="VS Code Command Palette: Dev Containers Reopen in Container" width="70%" />

Inside the Docker container, you can run torch-npu in the VS Code terminal:

<img src="./fig/vscode-terminal-pytorch-npu-smoke-test.png" alt="VS Code integrated terminal: PyTorch NPU smoke test" width="70%" />

## Step3: Enable Claude Code extension

Now, search for the "Claude Code for VS Code" extension and click **Install in Dev Container** (not **Install** on the host).

<img src="./fig/vscode-claude-code-extension-dev-container.png" alt="VS Code: Install Claude Code extension in Dev Container" width="100%" />

Open the Claude chat window and log in:

<img src="./fig/vscode-claude-code-auth-login.png" alt="VS Code: Claude Code sign-in" width="40%" />

In this environment, the Claude agent can execute code on the NPU.
