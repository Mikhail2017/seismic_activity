`--model` presets: `resnet18` (default), `resnet34`, `resnet34-horizon`,
`resnet50`, `efficientnet-b0`, `efficientnet-b4`, `vanilla`, or any SMP encoder
name. `resnet34-horizon` is ResNet34 plus the linear-moveout first-break prior
channel (5th input). A `*-horizon` suffix on any preset/SMP encoder does the
same. `--encoder-weights imagenet` (etc.) initializes the backbone; default is
train from scratch. `--model-config path.yaml` merges on top of the preset.
