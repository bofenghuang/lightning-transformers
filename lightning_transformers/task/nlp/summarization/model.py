from lightning_transformers.core.nlp.huggingface.seq2seq.model import HFSeq2SeqTransformer
from lightning_transformers.task.nlp.summarization.config import SummarizationTransformerConfig
from lightning_transformers.task.nlp.summarization.metric import RougeMetric


class SummarizationTransformer(HFSeq2SeqTransformer):
    cfg: SummarizationTransformerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bleu = None

    def compute_generate_metrics(self, batch, prefix):
        tgt_lns = self.tokenize_labels(batch["labels"])
        pred_lns = self.generate(batch["input_ids"], batch["attention_mask"])
        result = self.rouge(pred_lns, tgt_lns)
        self.log_dict(result, on_step=False, on_epoch=True)

    def configure_metrics(self, stage: str):
        self.rouge = RougeMetric(
            rouge_newline_sep=self.cfg.rouge_newline_sep,
            use_stemmer=self.cfg.use_stemmer,
        )

    @property
    def hf_pipeline_task(self) -> str:
        return "summarization"