import torch
import json
from kokoro import KModel
from loguru import logger
from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder
from kokoro.istftnet import Decoder
from transformers import AlbertConfig


class KModel1(torch.nn.Module):
    def __init__(self, config: str, model: str, disable_complex: bool = False):
        super().__init__()
        config = self.load_config(config)
        self.init_modules(config)
        self.load_model(model)
    
    def init_modules(self, config: dict):
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
    
    def load_config(self, config: str):
        if not isinstance(config, dict):
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        return config
    
    def load_model(self, model: str):
        for key, state_dict in torch.load(model, map_location='cpu', weights_only=True).items():
            if hasattr(self, key):
                try:
                    getattr(self, key).load_state_dict(state_dict)
                except:
                    logger.debug(f"Did not load {key} from state_dict")
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                    getattr(self, key).load_state_dict(state_dict, strict=False)
    
    def forward(self, input_ids: torch.LongTensor, ref_s: torch.Tensor, mask: torch.Tensor, speed: float = 1) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: compatible with different speed, make speed a tensor
        input_lengths = mask.sum(dim=1)
        bert_dur = self.bert(input_ids, attention_mask=(mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, ~mask)
        x, _ = self.predictor.lstm(d)
        t_en = self.text_encoder(input_ids, input_lengths, ~mask)
        
        
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1) / speed
        # size [bs, seq_len]
        pred_dur = torch.round(duration).clamp(min=1).long()

        return pred_dur, d, t_en


class KModel2(KModel1):
    def init_modules(self, config: dict):
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=False, **config['istftnet']
        )
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
    
    def generate_pred_aln_trg(self, indices: torch.Tensor, pred_dur: torch.Tensor, seq_len: int):
        pred_aln_trg = torch.zeros((seq_len, indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        return pred_aln_trg

    def forward(self, t_en: torch.Tensor, indices: torch.Tensor, pred_aln_trg: torch.Tensor, d: torch.Tensor, ref_s: torch.Tensor):
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        # [bs, 512, seq_len]
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        # audio shape = 600 * dur
        return audio


if __name__ == "__main__":
    config_path = "/app/api/src/models/v1_0/config.json"
    model_path = "/app/api/src/models/v1_0/kokoro-v1_0.pth"
    model1 = KModel1(config=config_path, model=model_path).eval()
    model2 = KModel2(config=config_path, model=model_path).eval()
    onnx_file1 = "kokoro/onnx/" + "kokoro1.onnx"
    onnx_file2 = "kokoro/onnx/" + "kokoro2.onnx"

    input_ids = torch.randint(1, 100, (48,)).numpy()
    input_ids = torch.LongTensor([[0, *input_ids, 0]])
    style = torch.randn(1, 256)
    speed = torch.randint(1, 10, (1,)).int()
    mask = torch.ones(1, 48 + 2).bool()

    if False:
        torch.onnx.export(
            model1, 
            args = (input_ids, style, mask, speed), 
            f = onnx_file1, 
            export_params = True, 
            verbose = True, 
            input_names = [ 'input_ids', 'style', 'mask', 'speed' ], 
            output_names = ['duration', 'd', 't_en'],
            opset_version = 17,
            dynamic_axes = {
                'input_ids': { 1: 'input_ids_len' },
                'mask': { 1: 'input_ids_len' },
                'duration': { 1: 'input_ids_len' }, 
                't_en': { 2: 'input_ids_len'},
                'd': { 1: 'input_ids_len'},
            }, 
            do_constant_folding = True, 
        )
    else:
        pred_dur = torch.sigmoid(torch.randn(1, 50, 64)).sum(-1).clamp(min=1).long()
        indices = torch.repeat_interleave(torch.arange(50), pred_dur[0])
        pred_aln_trg = model2.generate_pred_aln_trg(indices, pred_dur, 50)
        t_en = torch.randn(1, 512, 50)
        d = torch.randn(1, 50, 640)
        torch.onnx.export(
            model2, 
            args = (t_en, indices, pred_aln_trg, d, style), 
            f = onnx_file2, 
            export_params = True, 
            verbose = True, 
            input_names = [ 't_en', 'indices', 'pred_aln_trg', 'd', 'ref_s' ], 
            output_names = ['audio'],
            opset_version = 17,
            dynamic_axes = {
                't_en': { 2: 'input_ids_len' },
                'indices': { 0: 'dur_len' },
                'pred_aln_trg': { 1: 'input_ids_len', 2: 'dur_len' },
                'd': { 1: 'input_ids_len' },
            }, 
            do_constant_folding = True, 
        )
