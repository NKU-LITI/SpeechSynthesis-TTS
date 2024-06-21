import tensorflow as tf
from text import symbols

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    # 使用tf的方式定义超参数
    #hparams = tf.contrib.training.HParams( 
    hparams = AttrDict(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=375,  # 8*500=3000  # modify0
        iters_per_checkpoint=1000, # 1000, 每过xx个iter保存一下
        seed=1234, #随即种子
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        #ignore_layers=['embedding.weight'], # try2
        ignore_layers=['embedding.weight','encoder.'], # try1: 固定embed和encoder

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/vctk_p239_audio_text_train_filelist.txt',  # modify1
        validation_files='filelists/vctk_p239_audio_text_val_filelist.txt',  # modify2
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################ 
        max_wav_value=32768.0, # 音频的最大值
        sampling_rate=48000, #采样率, LJS是22050  #modify3
        filter_length=1024, # 计算梅尔频谱的滤波器的长度
        hop_length=256, # STFT的帧移
        win_length=1024, # STFT 的窗口长度
        n_mel_channels=80,
        mel_fmin=0.0, # tensorflow的版本修改这两了
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols), #26个英文字母+26个大写+逗号等符号
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3, #conv个数
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported，目前支持一帧一帧的预测也可以改为多帧
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5, #通过sigmod后如果大于0.5就停止，如果更严格可以变小
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=32, # 64, 显存爆了可以调小一点
        mask_padding=True  # set model's padded outputs to padded values
    )
    '''
    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())
    '''

    if hparams_string:
        hps = hparams_string[1:-2].split("-")
        for hp in hps:
            k,v = hp.split(":")
            if k in hparams:
                hparams[k] = v
                print("Set hparam: " + k + " to " + v)

    return hparams