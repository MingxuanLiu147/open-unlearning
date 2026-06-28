"""
IKE 知识编辑器（In-Context Knowledge Editing）
==============================================

基于上下文学习的知识编辑方法，不修改模型权重。
通过检索相似编辑事实作为 ICL 示例来实现知识编辑。

参考论文: Can We Edit Factual Knowledge by In-Context Learning?
https://arxiv.org/abs/2305.12740

核心思想：
1. 维护一个知识存储库（编辑事实的集合）
2. 使用 sentence-transformer 对编辑事实进行向量化
3. 推理时检索 top-k 最相似的事实作为 ICL 演示
4. 将演示前置到输入 prompt，引导模型输出新知识
"""

import logging
from typing import Optional, Dict, Any, List, Union

import torch

from trainer.edit.base import EditTrainer, EditRequest

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except ImportError:
    SentenceTransformer = None
    st_util = None


class IKEEditor(EditTrainer):
    """IKE 知识编辑器

    基于检索增强的上下文学习方法，不修改模型权重。
    维护编辑事实的向量索引，推理时检索最相关的事实作为 ICL 演示。

    Attributes:
        sentence_model_name: sentence-transformer 模型名称
        k: 检索的 ICL 示例数量
        use_icl_examples: 是否启用 ICL 示例检索
    """

    def __init__(
        self,
        sentence_model_name: str = "all-MiniLM-L6-v2",
        k: int = 3,
        use_icl_examples: bool = True,
        icl_prompt_template: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.sentence_model_name = sentence_model_name
        self.k = k
        self.use_icl_examples = use_icl_examples
        self.icl_prompt_template = icl_prompt_template or (
            "New Fact: {fact}\nPrompt: {prompt}\n\n"
        )

        self.knowledge_store: List[Dict[str, str]] = []
        self.knowledge_sentences: List[str] = []
        self.knowledge_embeddings: Optional[torch.Tensor] = None
        self._sentence_model: Optional[Any] = None

    @property
    def sentence_model(self):
        """延迟加载 sentence-transformer 模型。"""
        if self._sentence_model is not None:
            return self._sentence_model

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for IKE. "
                "Install it with: pip install sentence-transformers"
            )

        device = next(self.model.parameters()).device
        self._sentence_model = SentenceTransformer(self.sentence_model_name).to(device)
        logger.info(
            "Loaded sentence model '%s' on %s",
            self.sentence_model_name,
            device,
        )
        return self._sentence_model

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        """执行 IKE 知识编辑

        将编辑请求存入知识库并更新向量索引。
        IKE 不修改模型权重，编辑效果在推理时通过 ICL 实现。

        Args:
            requests: 编辑请求（单个或列表）

        Returns:
            包含编辑结果的字典：
            - success: 是否成功
            - edited_count: 成功编辑数
            - icl_examples: 为每个请求生成的 ICL 演示（供外部调用）
        """
        if isinstance(requests, EditRequest):
            requests = [requests]

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "icl_examples": [],
        }

        for request in requests:
            try:
                self._store_edit(request)
                results["edited_count"] += 1

                icl = self.get_icl_prompt(request.prompt) if self.knowledge_store else ""
                results["icl_examples"].append(icl)
            except Exception as e:
                logger.error(
                    "IKE edit failed for request: %s, error: %s",
                    request.prompt,
                    e,
                )
                results["success"] = False

        return results

    def _store_edit(self, request: EditRequest):
        """将单个编辑请求存入知识库并更新向量索引。"""
        fact = f"{request.prompt} {request.target_new}"
        entry = {
            "prompt": request.prompt,
            "subject": request.subject,
            "target_new": request.target_new,
            "target_old": request.target_old,
            "fact": fact,
        }
        self.knowledge_store.append(entry)

        sentence = self.icl_prompt_template.format(
            fact=fact, prompt=request.prompt
        )
        self.knowledge_sentences.append(sentence)

        self._update_embeddings(sentence)

        logger.info(
            "Stored edit: %s -> %s (store size: %d)",
            request.subject,
            request.target_new,
            len(self.knowledge_store),
        )

    def _update_embeddings(self, new_sentence: str):
        """增量更新知识库向量索引。"""
        if not self.use_icl_examples:
            return

        device = next(self.model.parameters()).device
        new_emb = self.sentence_model.encode(
            new_sentence, show_progress_bar=False, convert_to_tensor=True
        )
        if new_emb.dim() == 1:
            new_emb = new_emb.unsqueeze(0)
        new_emb = st_util.normalize_embeddings(new_emb.to(device))

        if self.knowledge_embeddings is None:
            self.knowledge_embeddings = new_emb
        else:
            self.knowledge_embeddings = torch.cat(
                [self.knowledge_embeddings, new_emb], dim=0
            )

    def get_icl_prompt(self, query_prompt: str) -> str:
        """根据查询 prompt 检索最相关的 ICL 演示

        Args:
            query_prompt: 用户查询 prompt

        Returns:
            拼接好的 ICL 演示字符串，可直接前置到模型输入
        """
        if (
            not self.use_icl_examples
            or not self.knowledge_store
            or self.knowledge_embeddings is None
        ):
            return ""

        device = self.knowledge_embeddings.device
        query_emb = self.sentence_model.encode(
            query_prompt, show_progress_bar=False, convert_to_tensor=True
        )
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        query_emb = st_util.normalize_embeddings(query_emb.to(device))

        top_k = min(self.k, len(self.knowledge_store))
        hits = st_util.semantic_search(
            query_emb,
            self.knowledge_embeddings,
            score_function=st_util.dot_score,
            top_k=top_k,
        )

        if not hits or not hits[0]:
            return ""

        icl_parts = [
            self.knowledge_sentences[hit["corpus_id"]] for hit in hits[0]
        ]
        return "".join(icl_parts)

    def rebuild_embeddings(self):
        """从头重建整个知识库的向量索引。

        当知识库被外部修改或需要更换 sentence model 时调用。
        """
        if not self.use_icl_examples or not self.knowledge_sentences:
            self.knowledge_embeddings = None
            return

        device = next(self.model.parameters()).device
        all_emb = self.sentence_model.encode(
            self.knowledge_sentences,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        self.knowledge_embeddings = st_util.normalize_embeddings(all_emb.to(device))
        logger.info(
            "Rebuilt embeddings for %d entries", len(self.knowledge_sentences)
        )

    def clear_knowledge_store(self):
        """清空知识库和向量索引。"""
        self.knowledge_store.clear()
        self.knowledge_sentences.clear()
        self.knowledge_embeddings = None
        logger.info("Knowledge store cleared")
