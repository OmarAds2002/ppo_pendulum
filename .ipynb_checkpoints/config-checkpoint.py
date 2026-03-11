import torch

class Config:
    ENV_NAME          = "Pendulum-v1"
    TOTAL_STEPS       = 500_000
    STEPS_PER_ROLLOUT = 2048
    UPDATE_EPOCHS     = 10
    MINIBATCH_SIZE    = 64
    GAMMA             = 0.99
    LAMBDA            = 0.95
    CLIP_EPS          = 0.2
    ENT_COEF          = 0.0
    VF_COEF           = 0.5
    LR                = 3e-4
    MAX_GRAD_NORM     = 0.5
    HIDDEN_SIZE       = 64
    DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
